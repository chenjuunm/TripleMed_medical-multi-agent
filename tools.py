"""Tool contracts and governance metadata.

Clinical data and knowledge tools are read-only. The demo examination-order
tool is an ``action`` and must receive approval bound to its exact proposal;
future write-capable tools must use the same clinician-approval boundary.
"""

import asyncio
import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from langchain_core.tools import tool

from rag_system import search_medical_guidelines


_DEMO_ORDER_STORE: Dict[str, Dict[str, Any]] = {}
_DEMO_ORDER_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ehr_source(patient_id: str, section: str) -> Dict[str, Any]:
    return {
        "type": "mock_ehr",
        "name": "演示 EHR",
        "record_id": f"{patient_id}:{section}",
        "retrieved_at": _now(),
        "is_demo": True,
    }


@tool
async def query_lab_results(patient_id: str, test_name: str) -> Dict[str, Any]:
    """读取患者已有实验室结果，不会创建或预约检查。"""

    await asyncio.sleep(0.1)
    mock_data = {
        "肌钙蛋白": "0.02 ng/mL（正常）",
        "血常规": "白细胞 11.2×10^9/L（偏高）",
        "D-二聚体": "0.4 mg/L（正常）",
    }
    found = test_name in mock_data
    return {
        "status": "found" if found else "not_found",
        "facts": [mock_data[test_name]] if found else [],
        "message": "" if found else f"未找到 {test_name} 的已有检查结果。",
        "source": _ehr_source(patient_id, f"lab:{test_name}"),
    }


@tool
async def get_patient_history(patient_id: str) -> Dict[str, Any]:
    """读取患者既往史、长期用药和过敏史。"""

    await asyncio.sleep(0.1)
    return {
        "status": "found",
        "facts": [
            "高血压病史 5 年",
            "规律服用氨氯地平",
            "青霉素过敏",
        ],
        "source": _ehr_source(patient_id, "history-medication-allergy"),
    }


@tool
async def check_drug_interaction(drug1: str, drug2: str) -> Dict[str, Any]:
    """只读检查两种药物的潜在相互作用，不执行处方或停药。"""

    await asyncio.sleep(0.2)
    normalized = {drug1.strip(), drug2.strip()}
    severe = {"阿司匹林", "华法林"}.issubset(normalized)
    statement = (
        "阿司匹林与华法林合用可显著增加出血风险。"
        if severe
        else "演示数据库未发现这两种药物的明确严重相互作用。"
    )
    return {
        "status": "found",
        "facts": [statement],
        "severity": "high" if severe else "none_detected",
        "source": {
            "type": "mock_drug_database",
            "name": "演示药物相互作用库",
            "record_id": f"interaction:{drug1}:{drug2}",
            "retrieved_at": _now(),
            "is_demo": True,
        },
    }


@tool
async def query_imaging_results(patient_id: str, imaging_type: str) -> Dict[str, Any]:
    """读取患者已有影像或心电报告，不会创建检查申请。"""

    await asyncio.sleep(0.1)
    found = imaging_type == "心电图"
    return {
        "status": "found" if found else "not_found",
        "facts": ["窦性心律，ST 段无明显压低或抬高。"] if found else [],
        "message": "" if found else f"未找到 {imaging_type} 的已有报告。",
        "source": _ehr_source(patient_id, f"imaging:{imaging_type}"),
    }


PATIENT_READ_TOOLS = [
    query_lab_results,
    get_patient_history,
    check_drug_interaction,
    query_imaging_results,
]
KNOWLEDGE_TOOLS = [search_medical_guidelines]


def _proposal_hash_payload(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Return exactly the immutable fields covered by clinician approval."""

    payload = {
        "proposal_id": proposal.get("proposal_id"),
        "case_id": proposal.get("case_id"),
        "patient_id": proposal.get("patient_id"),
        "version": proposal.get("version"),
        "items": proposal.get("items", []),
        "safety_flags": proposal.get("safety_flags", []),
        "created_at": proposal.get("created_at"),
        "expires_at": proposal.get("expires_at"),
    }
    # Bind retrieved-draft review to the exact report as well as any orders.
    for field in ("kind", "draft_hash", "knowledge_entry_id"):
        if field in proposal:
            payload[field] = proposal[field]
    return payload


def compute_exam_order_payload_hash(proposal: Dict[str, Any]) -> str:
    """Create a deterministic digest for the exact proposal shown to a clinician."""

    encoded = json.dumps(
        _proposal_hash_payload(proposal),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_exam_order_approval(
    proposal: Dict[str, Any], approval: Dict[str, Any]
) -> str:
    """Return an error message when an action lacks exact, current approval."""

    if proposal.get("status") != "pending_approval":
        return "检查申请不处于待审批状态。"
    if approval.get("decision") != "approved":
        return "医生未批准该检查申请。"
    for field in ("proposal_id", "case_id", "patient_id", "version", "payload_hash"):
        if approval.get(field) != proposal.get(field):
            return f"审批记录与检查申请的 {field} 不一致。"
    expected_hash = compute_exam_order_payload_hash(proposal)
    if proposal.get("payload_hash") != expected_hash:
        return "检查申请内容已在审批后发生变化。"
    expected_idempotency_key = (
        f"exam-order:{proposal.get('proposal_id')}:v{proposal.get('version')}:"
        f"{proposal.get('payload_hash')}"
    )
    if proposal.get("idempotency_key") != expected_idempotency_key:
        return "检查申请幂等键与已审批版本不一致。"
    try:
        if _parse_utc(proposal.get("expires_at")) <= datetime.now(timezone.utc):
            return "检查申请审批已过期。"
    except (TypeError, ValueError):
        return "检查申请缺少有效的过期时间。"
    if not str(approval.get("approver_id", "")).strip():
        return "审批记录缺少可信医生身份。"
    return ""


@tool
async def create_exam_order(
    proposal: Dict[str, Any], approval: Dict[str, Any]
) -> Dict[str, Any]:
    """Create one demo examination order after exact clinician approval."""

    validation_error = validate_exam_order_approval(proposal, approval)
    if validation_error:
        raise ValueError(validation_error)

    idempotency_key = str(proposal.get("idempotency_key", "")).strip()
    if not idempotency_key:
        raise ValueError("检查申请缺少幂等键。")

    with _DEMO_ORDER_LOCK:
        existing = _DEMO_ORDER_STORE.get(idempotency_key)
        if existing:
            return {**existing, "replayed": True, "message": "重复请求未再次开单。"}

        order_id = "DEMO-EXAM-" + hashlib.sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest()[:12].upper()
        result = {
            "status": "executed",
            "proposal_id": proposal.get("proposal_id"),
            "order_id": order_id,
            "idempotency_key": idempotency_key,
            "executed_at": _now(),
            "is_demo": True,
            "replayed": False,
            "message": "已创建演示检查申请；未写入真实医院系统。",
        }
        _DEMO_ORDER_STORE[idempotency_key] = dict(result)
        return result


ACTION_TOOLS = [create_exam_order]
medical_tools = PATIENT_READ_TOOLS + KNOWLEDGE_TOOLS + ACTION_TOOLS

TOOL_REGISTRY = {item.name: item for item in medical_tools}
TOOL_POLICIES: Dict[str, Dict[str, Any]] = {
    "query_lab_results": {
        "mode": "read_only",
        "data_scope": "patient_record",
        "requires_clinician_approval": False,
    },
    "get_patient_history": {
        "mode": "read_only",
        "data_scope": "patient_record",
        "requires_clinician_approval": False,
    },
    "query_imaging_results": {
        "mode": "read_only",
        "data_scope": "patient_record",
        "requires_clinician_approval": False,
    },
    "check_drug_interaction": {
        "mode": "read_only",
        "data_scope": "reference_calculation",
        "requires_clinician_approval": False,
    },
    "search_medical_guidelines": {
        "mode": "read_only",
        "data_scope": "medical_knowledge",
        "requires_clinician_approval": False,
    },
    "create_exam_order": {
        "mode": "action",
        "data_scope": "exam_order",
        "requires_clinician_approval": True,
    },
}


def is_allowed_read_tool(tool_name: str) -> bool:
    """Fail closed when a planner requests an unknown or action-capable tool."""

    policy = TOOL_POLICIES.get(tool_name, {})
    return policy.get("mode") == "read_only" and not policy.get(
        "requires_clinician_approval", True
    )
