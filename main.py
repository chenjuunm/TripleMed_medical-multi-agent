import json
import asyncio
import os
import secrets
import threading
import uuid

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Literal
import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
from langgraph.types import Command
from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator
from graph import medical_graph, _hard_safety_check, _present_context_text, validate_traceability
from answer_knowledge import AnswerKnowledgeBase, reusable_payload
from config import (
    AGENT_MODEL,
    ANSWER_KNOWLEDGE_DB,
    CLINICIAN_ALLOWED_ROLES,
    CLINICIAN_APPROVAL_TOKEN,
    MAX_REFLECTION_ROUNDS,
    MODEL_API_KEY,
    MODEL_BACKEND,
    MODEL_BASE_URL,
    MODEL_PREFLIGHT_TIMEOUT_SECONDS,
    ROUTER_MODEL,
    VERIFIER_MODEL,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI 临床决策支持系统 API")

# 配置 CORS (允许前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= Pydantic 模型定义 =================
class ChatRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)
    chief_complaint: str = Field(min_length=1, max_length=12000)
    # 显式 encounter/case_id 可避免同一患者多次就诊的状态相互污染。
    case_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    clinical_context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("patient_id", "chief_complaint", "case_id", mode="before")
    @classmethod
    def strip_nonempty_text(cls, value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空或仅包含空白字符")
        return cleaned


class FeedbackRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    patient_id: str = Field(min_length=1, max_length=128)
    vote: Literal["up", "down"]


class ConfirmRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)
    proposal_id: str = Field(min_length=1, max_length=128)
    proposal_version: int = Field(ge=1)
    payload_hash: str = Field(min_length=64, max_length=64)
    approved: StrictBool
    reason: Optional[str] = ""
    case_id: str = Field(min_length=1, max_length=128)

    @field_validator("patient_id", "case_id", "proposal_id", "payload_hash", mode="before")
    @classmethod
    def strip_nonempty_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空或仅包含空白字符")
        return cleaned

    @model_validator(mode="after")
    def require_rejection_reason(self):
        if not self.approved and not str(self.reason or "").strip():
            raise ValueError("拒绝检查申请时必须提供原因")
        return self


# ================= 全局变量 =================
# 原型阶段的 encounter -> thread 映射。生产环境应改为带 TTL 的持久会话存储。
knowledge_base = AnswerKnowledgeBase(ANSWER_KNOWLEDGE_DB)
case_threads: Dict[str, str] = {}
case_patients: Dict[str, str] = {}
approval_claims: Dict[tuple, str] = {}
approval_claims_lock = threading.Lock()

NODE_LABELS = {
    "reuse_liked_plan": "主诉高度相似，引用已点赞方案；本次未运行 Critic，等待医生审批",
    "intake_triage": "病例结构化与安全分流完成",
    "emergency_response": "已进入紧急分流路径",
    "supervisor": "临床任务规划完成",
    "patient_data": "患者记录读取完成",
    "knowledge_evidence": "医学证据检索完成",
    "specialist_panel": "独立临床视角分析完成",
    "synthesis": "证据融合与鉴别诊断完成",
    "critic": "独立证据与安全审校完成",
    "targeted_review": "定向补充检索完成",
    "finalize": "临床决策支持报告生成完成",
    "safe_finalize": "审校未通过，已安全升级人工复核",
    "prepare_exam_order": "检查申请草案生成完成",
    "await_clinician_approval": "医生审批完成",
    "execute_exam_order": "检查申请执行完成",
    "stop_without_execution": "检查申请未执行",
}


def authenticate_clinician(
    authorization: Optional[str], clinician_id: Optional[str], clinician_role: Optional[str]
) -> tuple[str, str]:
    """Authenticate the local HITL channel; production should use an IdP."""

    if not CLINICIAN_APPROVAL_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="医生审批通道未配置，请设置 CLINICIAN_APPROVAL_TOKEN。",
        )
    scheme, _, supplied_token = str(authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        supplied_token, CLINICIAN_APPROVAL_TOKEN
    ):
        raise HTTPException(status_code=401, detail="医生审批身份认证失败。")
    normalized_id = str(clinician_id or "").strip()
    normalized_role = str(clinician_role or "").strip().lower()
    if not normalized_id:
        raise HTTPException(status_code=401, detail="缺少医生身份标识。")
    if normalized_role not in CLINICIAN_ALLOWED_ROLES:
        raise HTTPException(status_code=403, detail="当前角色无权审批检查申请。")
    return normalized_id, normalized_role


async def lm_studio_runtime_status() -> Dict[str, Any]:
    """Check that every configured model is exposed by the active backend."""

    models_url = f"{MODEL_BASE_URL.rstrip('/')}/models"
    expected = {AGENT_MODEL, ROUTER_MODEL, VERIFIER_MODEL}
    headers = (
        {"Authorization": f"Bearer {MODEL_API_KEY}"}
        if MODEL_BACKEND == "remote_api"
        else None
    )
    try:
        async with httpx.AsyncClient(timeout=MODEL_PREFLIGHT_TIMEOUT_SECONDS) as client:
            response = await client.get(models_url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return {
            "ready": False,
            "backend": MODEL_BACKEND,
            "endpoint": models_url,
            "loaded_models": [],
            "missing_models": sorted(expected),
            "error": str(exc),
        }

    loaded = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }
    missing = expected - loaded
    return {
        "ready": not missing,
        "backend": MODEL_BACKEND,
        "endpoint": models_url,
        "loaded_models": sorted(loaded),
        "missing_models": sorted(missing),
        "error": "",
    }

# ================= 路由定义 =================

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """消除浏览器的 favicon.ico 404 警告"""
    return Response(status_code=204)


@app.get("/health")
async def health():
    model_status = await lm_studio_runtime_status()
    return {
        "status": "ready" if model_status["ready"] else "degraded",
        "models": model_status,
    }


@app.get("/")
async def serve_frontend():
    """提供前端 HTML 页面"""
    file_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="frontend.html 未找到，请确保它与 main.py 在同一目录下。")
    return FileResponse(file_path)


@app.post("/medical_chat")
async def medical_chat(request: ChatRequest):
    """处理医疗对话请求 (SSE 流式输出)"""
    if not medical_graph:
        raise HTTPException(status_code=500, detail="LangGraph 未正确初始化，请检查 graph.py。")

    references = []
    safety_text = "\n".join([request.chief_complaint, *_present_context_text(request.clinical_context)])
    # A similar liked plan can bypass models, never deterministic red flags or HITL.
    if not _hard_safety_check(safety_text)["red_flags"]:
        try:
            references = await asyncio.to_thread(knowledge_base.lookup, request.patient_id,
                                             request.chief_complaint, request.clinical_context)
        except Exception as exc:
            references = []
            logger.warning("问答知识库读取失败，继续常规工作流: %s", exc)
    if not references:
        model_status = await lm_studio_runtime_status()
        if not model_status["ready"]:
            missing = "、".join(model_status["missing_models"]) or "未知模型"
            backend_label = "远程模型 API" if MODEL_BACKEND == "remote_api" else "LM Studio"
            raise HTTPException(status_code=503,
                                detail=f"{backend_label} 尚未就绪，缺少可用模型：{missing}")

    # 每次分析必须有独立 case。旧客户端未提供时由服务端生成，绝不退化为 patient_id。
    case_key = request.case_id if request.case_id else str(uuid.uuid4())
    if case_key in case_threads:
        raise HTTPException(
            status_code=409,
            detail="case_id 已存在；新就诊请使用新的 case_id，审批请调用确认接口。",
        )
    thread_id = f"thread_{case_key}_{os.urandom(4).hex()}"
    case_threads[case_key] = thread_id
    case_patients[case_key] = request.patient_id

    config = {"configurable": {"thread_id": thread_id}}

    # 构建初始输入
    inputs = {
        "case_id": case_key,
        "patient_id": request.patient_id,
        "chief_complaint": request.chief_complaint,
        "clinical_context": request.clinical_context,
        "retrieved_plans": references,
        "messages": [],
        "evidence": [],
        "specialist_opinions": [],
        "audit_events": [],
        "revision_count": 0,
        "max_revision_rounds": MAX_REFLECTION_ROUNDS,
    }

    async def event_generator():
        try:
            session_data = {"type": "session", "case_id": case_key}
            yield f"data: {json.dumps(session_data, ensure_ascii=False)}\n\n"

            # 使用 astream 流式输出 LangGraph 的节点执行过程
            async for event in medical_graph.astream(inputs, config, stream_mode="updates"):
                if "__interrupt__" in event:
                    interrupt_info = event["__interrupt__"][0]
                    interrupt_value = interrupt_info.value or {}
                    interrupt_data = {
                        "type": "interrupt",
                        "data": {
                            "tool_name": interrupt_value.get("tool", "clinical_review"),
                            "tool_args": interrupt_value.get("args", {}),
                            "message": interrupt_value.get("message", "需要临床人员确认"),
                        },
                    }
                    yield f"data: {json.dumps(interrupt_data, ensure_ascii=False)}\n\n"
                    return  # 中断当前流，等待 /confirm_tool 恢复准确的草案版本。

                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        continue

                    status_data = {
                        "type": "status",
                        "node": node_name,
                        "message": NODE_LABELS.get(node_name, f"{node_name} 已完成"),
                    }
                    yield f"data: {json.dumps(status_data, ensure_ascii=False)}\n\n"

                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

                        # 发送流式数据给前端
                        data = {
                            "type": "message",
                            "node": node_name,
                            "content": content
                        }
                        if node_name in {"finalize", "safe_finalize", "emergency_response", "reuse_liked_plan"}:
                            data["feedback"] = {"case_id": case_key, "patient_id": request.patient_id}
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                    # 检查是否触发了 Interrupt (人工确认)
                    # LangGraph 的 interrupt 通常会在 __interrupt__ 键中
                    if "__interrupt__" in node_output:
                        interrupt_data = node_output["__interrupt__"]
                        # 提取工具调用信息 (根据您的 tools.py 调整)
                        tool_info = interrupt_data[0].value if isinstance(interrupt_data, list) else interrupt_data

                        data = {
                            "type": "interrupt",
                            "data": {
                                "tool_name": tool_info.get("tool", "unknown_tool"),
                                "tool_args": tool_info.get("args", {})
                            }
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                        return  # 遇到 interrupt 结束当前流，等待前端确认

            # 流结束标志
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"对话流处理出错: {e}", exc_info=True)
            error_data = {
                "type": "error",
                "message": "对话处理失败，请稍后重试。",
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/answer_feedback")
async def answer_feedback(request: FeedbackRequest):
    thread_id = case_threads.get(request.case_id)
    if not thread_id:
        raise HTTPException(status_code=404, detail="未找到该次问答。")
    if case_patients.get(request.case_id) != request.patient_id:
        raise HTTPException(status_code=403, detail="病例与患者不匹配。")
    snapshot = await medical_graph.aget_state({"configurable": {"thread_id": thread_id}})
    state = snapshot.values
    if not state.get("final_report"):
        raise HTTPException(status_code=409, detail="回答尚未完成。")
    try:
        if state.get("knowledge_hit", {}).get("mode") == "direct_plan":
            # Re-liking a retrieved plan records feedback on the same entry;
            # never turn an unreviewed current encounter into a new source plan.
            return await asyncio.to_thread(knowledge_base.feedback, request.case_id,
                request.patient_id, request.vote, state["knowledge_hit"]["entry_id"])
        payload = None
        if request.vote == "up":
            payload = reusable_payload(state)
            safety_text = "\n".join([state.get("chief_complaint", ""),
                                     *_present_context_text(state.get("clinical_context", {}))])
            if _hard_safety_check(safety_text)["red_flags"] or validate_traceability(payload):
                raise ValueError("存在安全标记或引用问题，不能作为共享参考。")
        return await asyncio.to_thread(
            knowledge_base.vote, request.case_id, request.patient_id,
            state["chief_complaint"], state.get("clinical_context", {}), request.vote, payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("保存问答反馈失败")
        raise HTTPException(status_code=503, detail="本地知识库暂时无法写入，请重试。") from exc


@app.post("/confirm_tool")
async def confirm_tool(
    request: ConfirmRequest,
    authorization: Optional[str] = Header(default=None),
    x_clinician_id: Optional[str] = Header(default=None),
    x_clinician_role: Optional[str] = Header(default=None),
):
    """Resume one exact examination proposal after trusted clinician review."""

    if not medical_graph:
        raise HTTPException(status_code=500, detail="LangGraph 未正确初始化。")
    clinician_id, clinician_role = authenticate_clinician(
        authorization, x_clinician_id, x_clinician_role
    )

    case_key = request.case_id
    thread_id = case_threads.get(case_key)
    if not thread_id:
        raise HTTPException(status_code=400, detail="未找到该病例会话，请重新开始问诊。")
    if case_patients.get(case_key) != request.patient_id:
        raise HTTPException(status_code=403, detail="病例会话与患者标识不匹配。")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await medical_graph.aget_state(config)
    except Exception as exc:
        logger.error("读取待审批状态失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="无法读取待审批工作流状态。") from exc

    values = snapshot.values if snapshot and isinstance(snapshot.values, dict) else {}
    proposal = values.get("exam_order_proposal", {})
    if not isinstance(proposal, dict) or proposal.get("status") != "pending_approval":
        raise HTTPException(status_code=409, detail="当前病例没有待审批检查申请。")

    expected = {
        "proposal_id": request.proposal_id,
        "case_id": request.case_id,
        "patient_id": request.patient_id,
        "version": request.proposal_version,
        "payload_hash": request.payload_hash,
    }
    if any(proposal.get(field) != value for field, value in expected.items()):
        raise HTTPException(
            status_code=409,
            detail="审批对象与当前检查申请版本不一致，已拒绝恢复执行。",
        )

    claim_key = (
        request.case_id,
        request.proposal_id,
        request.proposal_version,
        request.payload_hash,
    )
    decision_id = str(uuid.uuid4())
    with approval_claims_lock:
        if claim_key in approval_claims:
            raise HTTPException(status_code=409, detail="该检查申请已经提交过审批决定。")
        approval_claims[claim_key] = decision_id

    resume_value = {
        "decision_id": decision_id,
        "proposal_id": proposal.get("proposal_id"),
        "case_id": proposal.get("case_id"),
        "patient_id": proposal.get("patient_id"),
        "version": proposal.get("version"),
        "payload_hash": proposal.get("payload_hash"),
        "decision": "approved" if request.approved else "rejected",
        "approver_id": clinician_id,
        "approver_role": clinician_role,
        "reason": str(request.reason or "").strip(),
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }

    async def resume_generator():
        try:
            async for event in medical_graph.astream(
                Command(resume=resume_value), config, stream_mode="updates"
            ):
                if "__interrupt__" in event:
                    interrupt_info = event["__interrupt__"][0]
                    interrupt_value = interrupt_info.value or {}
                    interrupt_data = {
                        "type": "interrupt",
                        "data": {
                            "tool_name": interrupt_value.get("tool", "clinical_review"),
                            "tool_args": interrupt_value.get("args", {}),
                            "message": interrupt_value.get(
                                "message", "需要临床人员再次确认"
                            ),
                        },
                    }
                    yield f"data: {json.dumps(interrupt_data, ensure_ascii=False)}\n\n"
                    return

                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        continue
                    status_data = {
                        "type": "status",
                        "node": node_name,
                        "message": NODE_LABELS.get(node_name, f"{node_name} 已完成"),
                    }
                    yield f"data: {json.dumps(status_data, ensure_ascii=False)}\n\n"
                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        content = (
                            last_msg.content
                            if hasattr(last_msg, "content")
                            else str(last_msg)
                        )
                        data = {
                            "type": "message",
                            "node": node_name,
                            "content": content,
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("恢复对话流处理出错: %s", exc, exc_info=True)
            error_data = {
                "type": "error",
                "message": "审批恢复失败，检查申请未执行。",
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        resume_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
