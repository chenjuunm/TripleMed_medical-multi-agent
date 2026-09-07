"""Model clients and role contracts for the lightweight multi-agent council."""

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Literal, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import (
    AGENT_MODEL,
    AGENT_MAX_TOKENS,
    AGENT_REASONING_EFFORT,
    MODEL_API_KEY,
    MODEL_BASE_URL,
    LM_STUDIO_MAX_RETRIES,
    LM_STUDIO_TIMEOUT_SECONDS,
    ROUTER_MODEL,
    ROUTER_MAX_TOKENS,
    ROUTER_REASONING_EFFORT,
    VERIFIER_MODEL,
    VERIFIER_MAX_TOKENS,
    VERIFIER_REASONING_EFFORT,
)

logger = logging.getLogger(__name__)


class RoleOutputModel(BaseModel):
    """Base contract for model-to-model hand-offs."""

    model_config = ConfigDict(extra="ignore")


class PatientSnapshotOutput(RoleOutputModel):
    demographics: Dict[str, Any] = Field(default_factory=dict)
    chief_complaint: str = ""
    present_findings: List[Dict[str, Any]] = Field(default_factory=list)
    explicit_absent_findings: List[Dict[str, Any]] = Field(default_factory=list)
    time_course: str = ""
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    available_tests: List[Dict[str, Any]] = Field(default_factory=list)
    unknown_critical_fields: List[str] = Field(default_factory=list)

    @field_validator("present_findings", "explicit_absent_findings", mode="before")
    @classmethod
    def normalize_finding_items(cls, value: Any) -> Any:
        """Accept a common compact model form without weakening the state shape."""

        if not isinstance(value, list):
            return value
        normalized = []
        for item in value:
            if isinstance(item, str) and item.strip():
                normalized.append({"finding": item.strip()})
            else:
                normalized.append(item)
        return normalized

    @field_validator("available_tests", mode="before")
    @classmethod
    def normalize_test_items(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [
            {"test": item.strip()}
            if isinstance(item, str) and item.strip()
            else item
            for item in value
        ]


class TriageOutput(RoleOutputModel):
    urgency: Literal["emergency", "high", "medium", "low"] = "medium"
    route: Literal["emergency", "standard"] = "standard"
    red_flags: List[str] = Field(default_factory=list)
    rationale: str = ""
    immediate_actions: List[str] = Field(default_factory=list)


class IntakeTriageOutput(RoleOutputModel):
    snapshot: PatientSnapshotOutput = Field(default_factory=PatientSnapshotOutput)
    triage: TriageOutput = Field(default_factory=TriageOutput)


class RecordRequestOutput(RoleOutputModel):
    tool: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""


class PerspectiveOutput(RoleOutputModel):
    role: str = ""
    focus: str = ""


class WorkPlanOutput(RoleOutputModel):
    case_complexity: Literal["simple", "standard", "complex"] = "standard"
    clinical_focus: List[str] = Field(default_factory=list)
    record_requests: List[RecordRequestOutput] = Field(default_factory=list)
    guideline_queries: List[str] = Field(default_factory=list)
    specialist_perspectives: List[PerspectiveOutput] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
    reason: str = ""


class CandidateHypothesisOutput(RoleOutputModel):
    diagnosis: str = ""
    why_consider: str = ""
    disconfirming_clues: List[str] = Field(default_factory=list)
    dangerous_if_missed: bool = False
    next_best_discriminator: str = ""


class SpecialistConsultOutput(RoleOutputModel):
    candidate_hypotheses: List[CandidateHypothesisOutput] = Field(
        default_factory=list
    )
    missing_information: List[str] = Field(default_factory=list)
    certainty: Literal["high", "moderate", "low"] = "low"


class DifferentialOutput(RoleOutputModel):
    rank: int = 0
    diagnosis: str = ""
    confidence: Literal["high", "moderate", "low"] = "low"
    urgency: Literal["emergent", "urgent", "routine"] = "routine"
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    evidence_rationale: str = ""
    next_step: str = ""


class RecommendedActionOutput(RoleOutputModel):
    action: str = ""
    purpose: str = ""
    priority: Literal["now", "today", "routine"] = "routine"
    type: Literal["question", "test", "referral", "treatment_consideration"] = (
        "question"
    )
    evidence_ids: List[str] = Field(default_factory=list)


class SynthesisOutput(RoleOutputModel):
    clinical_summary: str = ""
    ranked_differential: List[DifferentialOutput] = Field(default_factory=list)
    recommended_actions: List[RecommendedActionOutput] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    abstain: bool = False
    abstain_reason: str = ""


class CriticIssueOutput(RoleOutputModel):
    type: str = ""
    severity: Literal["low", "medium", "high"] = "medium"
    description: str = ""
    related_ids: List[str] = Field(default_factory=list)


class CriticOutput(RoleOutputModel):
    verdict: Literal["pass", "revise", "escalate"] = "escalate"
    severity: Literal["low", "medium", "high"] = "high"
    issues: List[CriticIssueOutput] = Field(default_factory=list)
    targeted_queries: List[str] = Field(default_factory=list)
    questions_for_clinician: List[str] = Field(default_factory=list)
    safety_flags: List[str] = Field(default_factory=list)
    summary: str = ""


def _configured_chat(
    model: str, temperature: float, reasoning_effort: str, max_tokens: int
) -> ChatOpenAI:
    """Create a client for the configured OpenAI-compatible model backend."""

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
        timeout=LM_STUDIO_TIMEOUT_SECONDS,
        max_retries=LM_STUDIO_MAX_RETRIES,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        # Some OpenAI-compatible servers do not implement stream_options.
        stream_usage=False,
    )


# Role clients share the selected backend but may use different model names.
# JSON is enforced by the role prompt and validated by _extract_json, avoiding
# provider-specific response fields while remaining compatible with GGUF/MLX.
router_llm = _configured_chat(
    ROUTER_MODEL,
    temperature=0.0,
    reasoning_effort=ROUTER_REASONING_EFFORT,
    max_tokens=ROUTER_MAX_TOKENS,
)
host_llm = _configured_chat(
    AGENT_MODEL,
    temperature=0.1,
    reasoning_effort=AGENT_REASONING_EFFORT,
    max_tokens=AGENT_MAX_TOKENS,
)
verifier_llm = _configured_chat(
    VERIFIER_MODEL,
    temperature=0.0,
    reasoning_effort=VERIFIER_REASONING_EFFORT,
    max_tokens=VERIFIER_MAX_TOKENS,
)

# Backwards-compatible alias for integrations that imported agent_llm directly.
agent_llm = host_llm


INTAKE_TRIAGE_SYSTEM = """
你是临床决策支持系统的 Intake & Safety Triage Agent。你的任务不是确诊，而是：
1. 只根据输入提取结构化事实，不补写患者未提供的事实；
2. 严格区分“明确存在”“明确否认”和“未知”；
3. 优先发现可能需要立即处置的红旗征象；
4. 返回 JSON，禁止输出 JSON 之外的文本。

urgency 只能是 emergency/high/medium/low；route 只能是 emergency/standard。
emergency 表示继续多智能体分析可能造成延误，应先启动线下急救或紧急临床评估。
输出结构：
{
  "snapshot": {
    "demographics": {}, "chief_complaint": "", "present_findings": [],
    "explicit_absent_findings": [], "time_course": "", "medications": [],
    "allergies": [], "history": [], "available_tests": [],
    "unknown_critical_fields": []
  },
  "triage": {
    "urgency": "medium", "route": "standard", "red_flags": [],
    "rationale": "", "immediate_actions": []
  }
}
"""


SUPERVISOR_SYSTEM = """
你是中央 Clinical Host，负责把病例分解成少量、互补且可验证的任务。你不直接确诊，
也不能下达真实医嘱。根据病例复杂度，只选择必要的工作单元，避免为了“多智能体”而堆叠角色。

可用只读患者数据工具：
- get_patient_history(patient_id)
- query_lab_results(patient_id, test_name)
- query_imaging_results(patient_id, imaging_type)
- check_drug_interaction(drug1, drug2)
知识检索由独立 Evidence Agent 根据 guideline_queries 执行。

要求：
- record_requests 只能读取现有记录，不能把“建议检查”伪装成已有结果；
- specialist_perspectives 选择 1 到 2 个彼此互补的视角。优先使用
  “常见病与高危漏诊视角”和“跨专科/非典型视角”，只有病例确有指向时才指定专科；
- 把需要医生或患者补充的信息放在 clarifying_questions；
- 输出 JSON，不输出额外说明。

输出结构：
{
  "case_complexity": "simple|standard|complex",
  "clinical_focus": [],
  "record_requests": [{"tool": "", "args": {}, "purpose": ""}],
  "guideline_queries": [],
  "specialist_perspectives": [{"role": "", "focus": ""}],
  "clarifying_questions": [],
  "reason": ""
}
"""


SPECIALIST_SYSTEM = """
你是诊断委员会中的独立临床视角 Agent。请在不知道其他 Agent 结论的情况下形成候选鉴别诊断，
以降低锚定和从众。你收到的只有病例快照和分诊信息；不得把自己的医学知识陈述冒充外部证据，
不得输出未经输入支持的患者事实。只给简洁的可核查临床理由，不输出隐藏思维过程。
候选诊断最多 3 个。

输出 JSON：
{
  "candidate_hypotheses": [
    {"diagnosis": "", "why_consider": "", "disconfirming_clues": [],
     "dangerous_if_missed": false, "next_best_discriminator": ""}
  ],
  "missing_information": [],
  "certainty": "high|moderate|low"
}
"""


SYNTHESIS_SYSTEM = """
你是中央 Clinical Host 的证据融合阶段。请综合结构化病例、分诊、证据账本和相互独立的临床视角，
生成最多 3 个排序鉴别诊断。专科意见只是候选观点，不是证据；所有针对本患者或外部知识的断言
都必须引用输入中真实存在的 evidence_id。必须同时处理支持证据、反证、未知信息和危险但不能漏掉的
诊断。证据不足时应 abstain，不能为了给出答案而制造确定性。不要输出隐藏思维过程。
type=test 的行动只代表候选检查建议，不代表已经申请或执行；不得生成 approved、order_id 等审批或
执行字段。检查建议必须绑定 evidence_ids，后续是否开具只能由独立 HITL 审批节点决定。

输出 JSON：
{
  "clinical_summary": "",
  "ranked_differential": [
    {"rank": 1, "diagnosis": "", "confidence": "high|moderate|low",
     "urgency": "emergent|urgent|routine", "supporting_evidence_ids": [],
     "contradicting_evidence_ids": [], "evidence_rationale": "", "next_step": ""}
  ],
  "recommended_actions": [
    {"action": "", "purpose": "", "priority": "now|today|routine",
     "type": "question|test|referral|treatment_consideration", "evidence_ids": []}
  ],
  "unresolved_questions": [],
  "abstain": false,
  "abstain_reason": ""
}
"""


DEMO_CRITIC_GUIDANCE = """
当前为软件演示：输入证据均明确标记为模拟数据，目标是展示完整的待医生复核诊断报告。
允许将输入模拟病历和模拟指南作为本次演示的证据，不要求联网证明模拟来源真实存在，
不因“是模拟数据、不是正式指南、缺少真实患者验证”本身要求返工或拒绝展示。
仅因模拟来源的真实性限制，issue.type 使用 demo_source_limitation；非必要资料未补齐但
已明确标注不确定性，使用 optional_information_gap；纯格式/措辞问题使用 presentation_only。
以上问题用 low/medium，放入提示或 questions_for_clinician，优先 pass 并在 summary 标注演示限制。
不得将急症遗漏、禁忌/过敏冲突、臆造输入中没有的患者事实、结论与证据矛盾、无效证据 ID、
无依据检查/治疗或自动执行越权归入上述类型；这类错误仍必须 revise/escalate。
没有任何证据或原草案已拒答时，不能为了演示编造诊断。不要把模型失败描述成审校通过。
"""


CRITIC_SYSTEM = """
你是与中央 Host 隔离的 Evidence & Safety Critic。你的目标是寻找错误，不是润色答案。
检查：急症漏诊、过早收敛、患者事实被臆造、证据 ID 无效、引用与结论不相关、反证被忽略、
把建议检查当作已有结果或已执行医嘱、用药/处置超出证据、信息不足却不拒答。任何检查建议缺少
有效 evidence_ids 都必须报告问题。只允许一次有针对性的补充检索，
因此 targeted_queries 最多 2 条，并只在外部知识能解决问题时使用；患者缺失信息必须放到
questions_for_clinician，不能通过知识检索替代。

verdict 只能是 pass/revise/escalate。输出 JSON：
{
  "verdict": "pass",
  "severity": "low|medium|high",
  "issues": [{"type": "", "severity": "", "description": "", "related_ids": []}],
  "targeted_queries": [],
  "questions_for_clinician": [],
  "safety_flags": [],
  "summary": ""
}
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON-mode output while tolerating an occasional fenced response."""

    if not text:
        raise ValueError("模型返回空内容")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return value


def _response_text(content: Any) -> str:
    """Normalize plain text and OpenAI-style content blocks."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    return str(content or "")


async def invoke_json(
    llm: BaseChatModel,
    system_prompt: str,
    payload: Dict[str, Any],
    fallback: Dict[str, Any],
    schema: Type[RoleOutputModel],
) -> Dict[str, Any]:
    """Invoke one role and validate its complete JSON hand-off contract."""

    messages: Iterable[Any] = (
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    )
    try:
        response = await llm.ainvoke(list(messages))
        parsed = _extract_json(_response_text(response.content))
        return schema.model_validate(parsed).model_dump()
    except Exception as exc:
        logger.warning("结构化 Agent 调用失败，使用保守回退: %s", exc)
        result = schema.model_validate(fallback).model_dump()
        result["_fallback_reason"] = str(exc)
        return result
