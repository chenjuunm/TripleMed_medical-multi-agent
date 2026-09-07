"""Adaptive, evidence-centred multi-agent workflow.

The design is intentionally smaller than DeepRare: one central host, three
parallel evidence/opinion lanes, an independent critic and at most one targeted
reflection round. It is a clinician-facing decision-support workflow, not an
autonomous diagnosis or treatment system.
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Tuple

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from answer_knowledge import validate_plan

from agents import (
    DEMO_CRITIC_GUIDANCE,
    CRITIC_SYSTEM,
    CriticOutput,
    INTAKE_TRIAGE_SYSTEM,
    IntakeTriageOutput,
    SPECIALIST_SYSTEM,
    SYNTHESIS_SYSTEM,
    SpecialistConsultOutput,
    SUPERVISOR_SYSTEM,
    SynthesisOutput,
    WorkPlanOutput,
    host_llm,
    invoke_json,
    router_llm,
    verifier_llm,
)
from config import (
    CRITIC_DEMO_MODE,
    MAX_GUIDELINE_QUERIES,
    MAX_RECORD_REQUESTS,
    MAX_REFLECTION_ROUNDS,
    MAX_SPECIALIST_AGENTS,
    EXAM_ORDER_APPROVAL_TTL_SECONDS,
)
from state import EvidenceItem, MedicalState, SpecialistOpinion
from tools import (
    TOOL_POLICIES,
    TOOL_REGISTRY,
    compute_exam_order_payload_hash,
    create_exam_order,
    is_allowed_read_tool,
    search_medical_guidelines,
    validate_exam_order_approval,
)

logger = logging.getLogger(__name__)
memory = MemorySaver()


EMERGENCY_PHRASES = (
    "突发剧烈胸痛",
    "严重呼吸困难",
    "不能呼吸",
    "意识丧失",
    "昏迷",
    "口角歪斜",
    "单侧无力",
    "大出血",
    "呕血不止",
    "持续抽搐",
    "自杀计划",
)
URGENCY_RANK = {"low": 0, "medium": 1, "high": 2, "emergency": 3}
NEGATION_PREFIX = re.compile(
    r"(?:否认|未见|没有|并无|未出现|不存在|不伴|无)"
    r"(?:任何|明显|相关|上述|出现|发生|伴有|的|\s){0,8}$"
)
NEGATION_CHAIN_PREFIX = re.compile(
    r"(?:否认|未见|没有|并无|未出现|不存在|不伴|无)"
    r"(?:(?!但|然而|却|伴有|出现|存在).){0,24}"
    r"(?:及|和|与|、|或|以及|也不|也无)\s*$"
)
CLAUSE_BOUNDARY = re.compile(r"[，,。；;！？!?\n]")
ABSENCE_KEY_MARKERS = (
    "absent",
    "negative",
    "denied",
    "否认",
    "阴性",
    "不存在",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(agent: str, action: str, **details: Any) -> Dict[str, Any]:
    return {
        "timestamp": _now(),
        "agent": agent,
        "action": action,
        "details": details,
    }


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _phrase_is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 24) : start]
    clause_prefix = CLAUSE_BOUNDARY.split(prefix)[-1]
    return bool(
        NEGATION_PREFIX.search(clause_prefix)
        or NEGATION_CHAIN_PREFIX.search(clause_prefix)
    )


def _contains_non_negated_phrase(text: str, phrase: str) -> bool:
    return any(
        not _phrase_is_negated(text, match.start())
        for match in re.finditer(re.escape(phrase), text)
    )


def _present_context_text(value: Any, key: str = "") -> List[str]:
    """Flatten safety-relevant context while excluding explicitly absent data."""

    normalized_key = key.lower()
    if any(marker in normalized_key for marker in ABSENCE_KEY_MARKERS):
        return []
    if isinstance(value, dict):
        status = str(value.get("status", "")).lower()
        if status in {"absent", "negative", "denied", "明确否认"}:
            return []
        if value.get("present") is False:
            return []
        parts: List[str] = []
        for child_key, child_value in value.items():
            if str(child_key).lower() in {"status", "present"}:
                continue
            parts.extend(_present_context_text(child_value, str(child_key)))
        return parts
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.extend(_present_context_text(item, key))
        return parts
    if isinstance(value, (str, int, float)):
        return [str(value)]
    return []


def _hard_safety_check(text: str) -> Dict[str, Any]:
    """Small deterministic backstop; the LLM is not the sole emergency gate."""

    flags = [
        phrase
        for phrase in EMERGENCY_PHRASES
        if _contains_non_negated_phrase(text, phrase)
    ]
    urgency = "emergency" if flags else "low"

    blood_pressures = re.findall(r"(?<!\d)(\d{2,3})\s*/\s*(\d{2,3})(?!\d)", text)
    for systolic_raw, diastolic_raw in blood_pressures:
        systolic, diastolic = int(systolic_raw), int(diastolic_raw)
        if systolic >= 180 or diastolic >= 120:
            bp_flag = f"重度血压升高 {systolic}/{diastolic} mmHg"
            flags.append(bp_flag)
            end_organ_symptoms = (
                "胸痛",
                "呼吸困难",
                "意识障碍",
                "意识模糊",
                "意识丧失",
                "偏瘫",
                "单侧无力",
                "视物不清",
                "剧烈头痛",
            )
            urgency = (
                "emergency"
                if any(
                    _contains_non_negated_phrase(text, symptom)
                    for symptom in end_organ_symptoms
                )
                else "high"
            )

    return {"urgency": urgency, "red_flags": list(dict.fromkeys(flags))}


async def intake_triage_node(state: MedicalState) -> Dict[str, Any]:
    chief_complaint = state.get("chief_complaint", "").strip()
    clinical_context = state.get("clinical_context", {})
    fallback = {
        "snapshot": {
            "demographics": clinical_context.get("demographics", {}),
            "chief_complaint": chief_complaint,
            "present_findings": [
                {"finding": chief_complaint, "source": "user"}
            ]
            if chief_complaint
            else [],
            "explicit_absent_findings": [],
            "time_course": "",
            "medications": [],
            "allergies": [],
            "history": [],
            "available_tests": [],
            "unknown_critical_fields": ["年龄", "生命体征", "查体"],
        },
        "triage": {
            "urgency": "medium",
            "route": "standard",
            "red_flags": [],
            "rationale": "结构化分诊不可用，采用保守默认分级。",
            "immediate_actions": [],
        },
    }
    result = await invoke_json(
        router_llm,
        INTAKE_TRIAGE_SYSTEM,
        {
            "chief_complaint": chief_complaint,
            "clinical_context": clinical_context,
        },
        fallback,
        IntakeTriageOutput,
    )

    snapshot = result.get("snapshot") if isinstance(result.get("snapshot"), dict) else {}
    snapshot = {**fallback["snapshot"], **snapshot}
    snapshot["chief_complaint"] = chief_complaint

    triage = result.get("triage") if isinstance(result.get("triage"), dict) else {}
    triage = {**fallback["triage"], **triage}
    if triage.get("urgency") not in URGENCY_RANK:
        triage["urgency"] = "medium"

    safety_parts = [chief_complaint]
    safety_parts.extend(_present_context_text(clinical_context))
    deterministic = _hard_safety_check("\n".join(safety_parts))
    if URGENCY_RANK[deterministic["urgency"]] > URGENCY_RANK[triage["urgency"]]:
        triage["urgency"] = deterministic["urgency"]
        triage["rationale"] = (
            f"确定性安全规则升级分级；原模型说明：{triage.get('rationale', '')}"
        )
    model_red_flags = [
        str(item).strip()
        for item in _as_list(triage.get("red_flags"))
        if isinstance(item, (str, int, float)) and str(item).strip()
    ]
    triage["red_flags"] = list(
        dict.fromkeys(model_red_flags + deterministic["red_flags"])
    )
    triage["immediate_actions"] = [
        str(item).strip()
        for item in _as_list(triage.get("immediate_actions"))
        if isinstance(item, (str, int, float)) and str(item).strip()
    ]
    triage["route"] = (
        "emergency" if triage["urgency"] == "emergency" else "standard"
    )

    return {
        "case_snapshot": snapshot,
        "triage_result": triage,
        "requires_clinician_review": triage["urgency"] in {"emergency", "high"},
        "audit_events": [
            _event(
                "IntakeTriageAgent",
                "case_structured_and_triaged",
                urgency=triage["urgency"],
                route=triage["route"],
                red_flags=triage["red_flags"],
            )
        ],
    }


def route_after_triage(state: MedicalState) -> Literal["emergency", "standard"]:
    return (
        "emergency"
        if state.get("triage_result", {}).get("route") == "emergency"
        else "standard"
    )


async def emergency_response_node(state: MedicalState) -> Dict[str, Any]:
    triage = state.get("triage_result", {})
    flags = "、".join(_as_list(triage.get("red_flags"))) or "存在潜在急症信号"
    # Never render model-generated treatment text on the pre-Critic emergency
    # path. Only this fixed, clinician-reviewed escalation language is allowed.
    model_actions_suppressed = len(_as_list(triage.get("immediate_actions")))
    action_lines = (
        "- 立即由具备资质的临床人员进行面对面评估，并按本机构急救流程处置。\n"
        "- 如患者不在医疗机构内，请联系当地急救服务。\n"
        "- 不应等待本系统完成常规鉴别诊断后再升级处置。"
    )
    report = f"""## 紧急分流提示

检测到红旗征象：**{flags}**。

{action_lines}

本次工作流已停止常规多智能体分析，避免自动化流程延误紧急处置。本提示是临床决策支持信息，不能替代急救流程或医生判断。
"""
    return {
        "final_report": report,
        "requires_clinician_review": True,
        "messages": [AIMessage(content=report, name="EmergencySafetyAgent")],
        "audit_events": [
            _event(
                "EmergencySafetyAgent",
                "routine_workflow_stopped",
                red_flags=flags,
                model_actions_suppressed=model_actions_suppressed,
            )
        ],
    }


def _default_perspectives(complexity: str) -> List[Dict[str, str]]:
    perspectives = [
        {
            "role": "常见病与高危漏诊视角",
            "focus": "覆盖常见解释，同时优先识别时间敏感且不能漏掉的诊断。",
        }
    ]
    if complexity != "simple" and MAX_SPECIALIST_AGENTS > 1:
        perspectives.append(
            {
                "role": "跨专科与非典型表现视角",
                "focus": "寻找不能被首要假设解释的线索、跨系统疾病和非典型表现。",
            }
        )
    return perspectives


def _sanitize_plan(raw: Dict[str, Any], state: MedicalState) -> Dict[str, Any]:
    complexity = raw.get("case_complexity", "standard")
    if complexity not in {"simple", "standard", "complex"}:
        complexity = "standard"

    patient_id = state.get("patient_id", "")
    requests: List[Dict[str, Any]] = []
    for item in _as_list(raw.get("record_requests")):
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool")
        if tool_name not in {
            "get_patient_history",
            "query_lab_results",
            "query_imaging_results",
            "check_drug_interaction",
        }:
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        args = dict(args)
        if tool_name != "check_drug_interaction":
            args["patient_id"] = patient_id
        requests.append(
            {
                "tool": tool_name,
                "args": args,
                "purpose": str(item.get("purpose", "读取相关患者记录")),
            }
        )
        if len(requests) >= MAX_RECORD_REQUESTS:
            break
    if not requests:
        requests = [
            {
                "tool": "get_patient_history",
                "args": {"patient_id": patient_id},
                "purpose": "核对既往史、长期用药和过敏史",
            }
        ]

    guideline_queries = [
        str(query).strip()
        for query in _as_list(raw.get("guideline_queries"))
        if str(query).strip()
    ][:MAX_GUIDELINE_QUERIES]
    if not guideline_queries:
        guideline_queries = [
            f"{state.get('chief_complaint', '')} 鉴别诊断 临床指南"
        ]

    requested_perspectives = [
        item
        for item in _as_list(raw.get("specialist_perspectives"))
        if isinstance(item, dict) and str(item.get("role", "")).strip()
    ]
    perspective_limit = 1 if complexity == "simple" else MAX_SPECIALIST_AGENTS
    perspectives = requested_perspectives[:perspective_limit]
    if not perspectives:
        perspectives = _default_perspectives(complexity)[:perspective_limit]

    return {
        "case_complexity": complexity,
        "clinical_focus": _as_list(raw.get("clinical_focus")),
        "record_requests": requests,
        "guideline_queries": guideline_queries,
        "specialist_perspectives": [
            {
                "role": str(item.get("role", "独立临床视角")),
                "focus": str(item.get("focus", "形成独立鉴别诊断")),
            }
            for item in perspectives
        ],
        "clarifying_questions": [
            str(item)
            for item in _as_list(raw.get("clarifying_questions"))
            if str(item).strip()
        ],
        "reason": str(raw.get("reason", "")),
    }


async def supervisor_node(state: MedicalState) -> Dict[str, Any]:
    fallback = {
        "case_complexity": "standard",
        "clinical_focus": [],
        "record_requests": [],
        "guideline_queries": [],
        "specialist_perspectives": [],
        "clarifying_questions": state.get("case_snapshot", {}).get(
            "unknown_critical_fields", []
        ),
        "reason": "采用保守默认任务分解。",
    }
    raw_plan = await invoke_json(
        host_llm,
        SUPERVISOR_SYSTEM,
        {
            "patient_id": state.get("patient_id"),
            "case_snapshot": state.get("case_snapshot", {}),
            "triage": state.get("triage_result", {}),
            "budgets": {
                "record_requests": MAX_RECORD_REQUESTS,
                "guideline_queries": MAX_GUIDELINE_QUERIES,
                "specialist_agents": MAX_SPECIALIST_AGENTS,
                "reflection_rounds": state.get(
                    "max_revision_rounds", MAX_REFLECTION_ROUNDS
                ),
            },
        },
        fallback,
        WorkPlanOutput,
    )
    plan = _sanitize_plan(raw_plan, state)
    return {
        "work_plan": plan,
        "plan_result": plan,
        "audit_events": [
            _event(
                "ClinicalHost",
                "work_plan_created",
                complexity=plan["case_complexity"],
                record_requests=len(plan["record_requests"]),
                guideline_queries=len(plan["guideline_queries"]),
                specialist_agents=len(plan["specialist_perspectives"]),
            )
        ],
    }


def _tool_result_to_evidence(
    evidence_id: str,
    tool_name: str,
    args: Dict[str, Any],
    purpose: str,
    result: Any,
) -> EvidenceItem:
    result_dict = result if isinstance(result, dict) else {"facts": [_safe_text(result)]}
    facts = [str(item) for item in _as_list(result_dict.get("facts"))]
    statement = "；".join(facts) or str(
        result_dict.get("message", "工具未返回可用事实。")
    )
    source = result_dict.get("source") if isinstance(result_dict.get("source"), dict) else {}
    kind = "patient_record"
    if tool_name == "check_drug_interaction":
        kind = "drug_safety_reference"
    elif result_dict.get("status") != "found":
        kind = "record_gap"
    return {
        "evidence_id": evidence_id,
        "kind": kind,
        "statement": statement,
        "source": source,
        "query": {"tool": tool_name, "args": args},
        "purpose": purpose,
        "reliability": (
            "primary_patient_record"
            if kind == "patient_record"
            else "reference_or_record_status"
        ),
        "agent": "PatientDataAgent",
        "retrieved_at": source.get("retrieved_at", _now()),
    }


async def patient_data_node(state: MedicalState) -> Dict[str, Any]:
    requests = _as_list(state.get("work_plan", {}).get("record_requests"))

    async def execute(item: Dict[str, Any], index: int) -> Tuple[Any, Dict[str, Any]]:
        tool_name = str(item.get("tool", ""))
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        purpose = str(item.get("purpose", ""))
        if not is_allowed_read_tool(tool_name) or tool_name not in TOOL_REGISTRY:
            return None, _event(
                "PatientDataAgent",
                "tool_blocked_by_policy",
                tool=tool_name,
                policy=TOOL_POLICIES.get(tool_name),
            )
        try:
            raw_result = await TOOL_REGISTRY[tool_name].ainvoke(args)
            evidence = _tool_result_to_evidence(
                f"P{index + 1:03d}", tool_name, args, purpose, raw_result
            )
            return evidence, _event(
                "PatientDataAgent",
                "read_tool_completed",
                tool=tool_name,
                evidence_id=evidence["evidence_id"],
            )
        except Exception as exc:
            return None, _event(
                "PatientDataAgent",
                "read_tool_failed",
                tool=tool_name,
                error=str(exc),
            )

    pairs = await asyncio.gather(
        *(execute(item, index) for index, item in enumerate(requests))
    )
    evidence = [item for item, _ in pairs if item]
    events = [event for _, event in pairs]
    return {"evidence": evidence, "audit_events": events}


async def _retrieve_knowledge(
    queries: List[str], prefix: str, agent_name: str
) -> Tuple[List[EvidenceItem], List[Dict[str, Any]]]:
    async def search(query: str) -> Tuple[str, Any]:
        try:
            return query, await search_medical_guidelines.ainvoke({"query": query})
        except Exception as exc:
            return query, {"status": "error", "items": [], "message": str(exc)}

    raw_results = await asyncio.gather(*(search(query) for query in queries))
    pending: List[Tuple[str, Dict[str, Any]]] = []
    events: List[Dict[str, Any]] = []
    for query, result in raw_results:
        result_dict = result if isinstance(result, dict) else {}
        items = [item for item in _as_list(result_dict.get("items")) if isinstance(item, dict)]
        if not items:
            events.append(
                _event(
                    agent_name,
                    "knowledge_query_no_evidence",
                    query=query,
                    status=result_dict.get("status", "unknown"),
                    message=result_dict.get("message", ""),
                )
            )
        for item in items:
            pending.append((query, item))

    evidence: List[EvidenceItem] = []
    for index, (query, item) in enumerate(pending):
        locator = item.get("source_url") or item.get("source_id")
        if not locator:
            events.append(
                _event(
                    agent_name,
                    "untraceable_knowledge_item_rejected",
                    query=query,
                    source=item.get("source"),
                )
            )
            continue
        evidence_id = f"{prefix}{index + 1:03d}"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "medical_knowledge",
                "statement": str(item.get("content", "")),
                "source": {
                    "type": "guideline_or_consensus",
                    "name": item.get("source", "未知来源"),
                    "source_id": item.get("source_id"),
                    "source_url": item.get("source_url"),
                    "version": item.get("version"),
                    "is_demo": item.get("is_demo", False),
                },
                "query": {"query": query},
                "purpose": "为鉴别诊断和下一步行动提供可追溯外部依据",
                "reliability": "retrieved_medical_source",
                "agent": agent_name,
                "retrieved_at": item.get("retrieved_at", _now()),
            }
        )
        events.append(
            _event(
                agent_name,
                "traceable_knowledge_item_added",
                evidence_id=evidence_id,
                locator=locator,
            )
        )
    return evidence, events


async def knowledge_evidence_node(state: MedicalState) -> Dict[str, Any]:
    queries = [
        str(item)
        for item in _as_list(state.get("work_plan", {}).get("guideline_queries"))
    ][:MAX_GUIDELINE_QUERIES]
    evidence, events = await _retrieve_knowledge(queries, "K", "EvidenceAgent")
    return {"evidence": evidence, "audit_events": events}


async def specialist_panel_node(state: MedicalState) -> Dict[str, Any]:
    perspectives = _as_list(
        state.get("work_plan", {}).get("specialist_perspectives")
    )[:MAX_SPECIALIST_AGENTS]

    async def consult(item: Dict[str, Any], index: int) -> SpecialistOpinion:
        role = str(item.get("role", "独立临床视角"))
        focus = str(item.get("focus", "形成独立鉴别诊断"))
        result = await invoke_json(
            host_llm,
            SPECIALIST_SYSTEM,
            {
                "assigned_role": role,
                "assigned_focus": focus,
                "case_snapshot": state.get("case_snapshot", {}),
                "triage": state.get("triage_result", {}),
            },
            {
                "candidate_hypotheses": [],
                "missing_information": [],
                "certainty": "low",
            },
            SpecialistConsultOutput,
        )
        return {
            "opinion_id": f"O{index + 1:03d}",
            "role": role,
            "focus": focus,
            "candidate_hypotheses": _as_list(result.get("candidate_hypotheses"))[:5],
            "missing_information": [
                str(value) for value in _as_list(result.get("missing_information"))
            ],
            "certainty": str(result.get("certainty", "low")),
        }

    opinions = await asyncio.gather(
        *(consult(item, index) for index, item in enumerate(perspectives))
    )
    return {
        "specialist_opinions": list(opinions),
        "audit_events": [
            _event(
                "SpecialistPanel",
                "independent_consultations_completed",
                opinion_ids=[item["opinion_id"] for item in opinions],
            )
        ],
    }


def _normalize_synthesis(raw: Dict[str, Any], state: MedicalState) -> Dict[str, Any]:
    differentials = [
        item
        for item in _as_list(raw.get("ranked_differential"))
        if isinstance(item, dict) and str(item.get("diagnosis", "")).strip()
    ][:3]
    for index, item in enumerate(differentials):
        item["rank"] = index + 1
        item["supporting_evidence_ids"] = [
            str(value) for value in _as_list(item.get("supporting_evidence_ids"))
        ]
        item["contradicting_evidence_ids"] = [
            str(value) for value in _as_list(item.get("contradicting_evidence_ids"))
        ]

    actions = [
        item
        for item in _as_list(raw.get("recommended_actions"))
        if isinstance(item, dict) and str(item.get("action", "")).strip()
    ][:6]
    for item in actions:
        item["evidence_ids"] = [
            str(value) for value in _as_list(item.get("evidence_ids"))
        ]

    unresolved = [
        str(value)
        for value in _as_list(raw.get("unresolved_questions"))
        if str(value).strip()
    ]
    unresolved.extend(
        str(value)
        for value in _as_list(
            state.get("work_plan", {}).get("clarifying_questions")
        )
        if str(value).strip() and str(value) not in unresolved
    )
    return {
        "clinical_summary": str(raw.get("clinical_summary", "")),
        "ranked_differential": differentials,
        "recommended_actions": actions,
        "unresolved_questions": unresolved,
        "abstain": bool(raw.get("abstain", not differentials)),
        "abstain_reason": str(raw.get("abstain_reason", "")),
    }


async def synthesis_node(state: MedicalState) -> Dict[str, Any]:
    fallback = {
        "clinical_summary": "当前可追溯证据不足，无法形成可靠的排序鉴别诊断。",
        "ranked_differential": [],
        "recommended_actions": [],
        "unresolved_questions": state.get("case_snapshot", {}).get(
            "unknown_critical_fields", []
        ),
        "abstain": True,
        "abstain_reason": "证据融合模型不可用或信息不足。",
    }
    raw = await invoke_json(
        host_llm,
        SYNTHESIS_SYSTEM,
        {
            "case_snapshot": state.get("case_snapshot", {}),
            "triage": state.get("triage_result", {}),
            "evidence_ledger": state.get("evidence", []),
            "independent_opinions": state.get("specialist_opinions", []),
            "revision_round": state.get("revision_count", 0),
        },
        fallback,
        SynthesisOutput,
    )
    synthesis = _normalize_synthesis(raw, state)
    return {
        "synthesis_result": synthesis,
        "reasoning_result": synthesis,
        "audit_events": [
            _event(
                "ClinicalHost",
                "evidence_synthesized",
                hypotheses=len(synthesis["ranked_differential"]),
                abstain=synthesis["abstain"],
                revision_round=state.get("revision_count", 0),
            )
        ],
    }


def _source_locator(source: Dict[str, Any]) -> Any:
    return (
        source.get("source_url")
        or source.get("source_id")
        or source.get("record_id")
    )


def validate_traceability(state: MedicalState) -> List[Dict[str, Any]]:
    evidence = state.get("evidence", [])
    evidence_by_id = {
        item.get("evidence_id"): item
        for item in evidence
        if isinstance(item, dict) and item.get("evidence_id")
    }
    valid_ids = {
        item.get("evidence_id")
        for item in evidence
        if isinstance(item, dict) and item.get("evidence_id")
    }
    issues: List[Dict[str, Any]] = []

    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        if not _source_locator(source):
            issues.append(
                {
                    "type": "missing_provenance",
                    "severity": "high",
                    "description": f"证据 {item.get('evidence_id')} 缺少可定位来源。",
                    "related_ids": [item.get("evidence_id")],
                }
            )
        source_url = source.get("source_url")
        if source_url and not str(source_url).startswith(("https://", "http://")):
            issues.append(
                {
                    "type": "invalid_source_url",
                    "severity": "high",
                    "description": f"证据 {item.get('evidence_id')} 的来源链接格式无效。",
                    "related_ids": [item.get("evidence_id")],
                }
            )

    synthesis = state.get("synthesis_result", {})
    for hypothesis in _as_list(synthesis.get("ranked_differential")):
        if not isinstance(hypothesis, dict):
            continue
        diagnosis = hypothesis.get("diagnosis", "未命名诊断")
        cited = _as_list(hypothesis.get("supporting_evidence_ids")) + _as_list(
            hypothesis.get("contradicting_evidence_ids")
        )
        unknown = [item for item in cited if item not in valid_ids]
        if unknown:
            issues.append(
                {
                    "type": "unknown_evidence_id",
                    "severity": "high",
                    "description": f"{diagnosis} 引用了不存在的证据 ID：{unknown}。",
                    "related_ids": unknown,
                }
            )
        if not _as_list(hypothesis.get("supporting_evidence_ids")):
            issues.append(
                {
                    "type": "unsupported_hypothesis",
                    "severity": "medium",
                    "description": f"{diagnosis} 没有绑定支持证据。",
                    "related_ids": [],
                }
            )

    for action in _as_list(synthesis.get("recommended_actions")):
        if not isinstance(action, dict):
            continue
        cited = _as_list(action.get("evidence_ids"))
        unknown = [item for item in cited if item not in valid_ids]
        if unknown:
            issues.append(
                {
                    "type": "untraceable_action",
                    "severity": "high",
                    "description": f"建议行动引用了不存在的证据 ID：{unknown}。",
                    "related_ids": unknown,
                }
            )
        if action.get("type") in {"test", "referral"} and not cited:
            issues.append(
                {
                    "type": "unsupported_clinical_action",
                    "severity": "high",
                    "description": "检查或转诊建议缺少可追溯依据。",
                    "related_ids": [],
                }
            )
        if action.get("type") == "treatment_consideration" and not cited:
            issues.append(
                {
                    "type": "unsupported_treatment",
                    "severity": "high",
                    "description": "治疗类建议缺少可追溯依据。",
                    "related_ids": [],
                }
            )
        if action.get("type") == "treatment_consideration" and cited:
            cited_kinds = {
                evidence_by_id[item].get("kind")
                for item in cited
                if item in evidence_by_id
            }
            if not cited_kinds.intersection(
                {"medical_knowledge", "drug_safety_reference"}
            ):
                issues.append(
                    {
                        "type": "treatment_without_external_basis",
                        "severity": "high",
                        "description": "治疗类建议仅引用了患者事实，缺少指南或药物安全来源。",
                        "related_ids": cited,
                    }
                )
    return issues


async def critic_node(state: MedicalState) -> Dict[str, Any]:
    deterministic_issues = validate_traceability(state)
    no_evidence = not state.get("evidence")
    demo_mode = bool(CRITIC_DEMO_MODE and not no_evidence and all(
        isinstance(item, dict) and isinstance(item.get("source"), dict)
        and item["source"].get("is_demo") is True for item in state["evidence"]
    ))
    synthesis = state.get("synthesis_result", {})
    fallback_verdict = "escalate" if no_evidence or synthesis.get("abstain") else (
        "revise" if deterministic_issues else "pass"
    )
    raw = await invoke_json(
        verifier_llm,
        CRITIC_SYSTEM + (DEMO_CRITIC_GUIDANCE if demo_mode else ""),
        {
            "case_snapshot": state.get("case_snapshot", {}),
            "triage": state.get("triage_result", {}),
            "evidence_ledger": state.get("evidence", []),
            "host_synthesis": synthesis,
            "review_mode": "demo_presentation" if demo_mode else "standard",
            "deterministic_traceability_findings": deterministic_issues,
            "remaining_reflection_rounds": max(
                0,
                state.get("max_revision_rounds", MAX_REFLECTION_ROUNDS)
                - state.get("revision_count", 0),
            ),
        },
        {
            "verdict": fallback_verdict,
            "severity": "high" if fallback_verdict == "escalate" else "medium",
            "issues": [],
            "targeted_queries": [],
            "questions_for_clinician": [],
            "safety_flags": [],
            "summary": "审校模型不可用，依据确定性检查采取保守结论。",
        },
        CriticOutput,
    )

    verdict = raw.get("verdict", fallback_verdict)
    if verdict not in {"pass", "revise", "escalate"}:
        verdict = fallback_verdict
    model_issues = [
        item for item in _as_list(raw.get("issues")) if isinstance(item, dict)
    ]
    advisory_types = {"demo_source_limitation", "optional_information_gap", "presentation_only"}
    warnings = [item for item in model_issues if demo_mode
                and item.get("type") in advisory_types and item.get("severity") in {"low", "medium"}]
    issues = deterministic_issues + [item for item in model_issues if item not in warnings]
    # Relax only explicit presentation limitations, not all low/medium issues.
    if (warnings and not issues and verdict in {"pass", "revise"}
            and raw.get("severity") in {"low", "medium"} and not raw.get("safety_flags")):
        verdict = "pass"
    if no_evidence or synthesis.get("abstain"):
        verdict = "escalate"
    elif issues and verdict == "pass":
        # Reported safety/provenance findings are not advisory; an internally
        # inconsistent model response cannot carry a bare "pass" verdict.
        verdict = "revise"

    if "_fallback_reason" in raw or (demo_mode and (
            raw.get("safety_flags") or raw.get("severity") == "high")):
        verdict = "escalate"

    critique = {
        "verdict": verdict,
        "severity": str(raw.get("severity", "medium")),
        "issues": issues,
        "warnings": warnings,
        "review_mode": "demo_presentation" if demo_mode else "standard",
        "targeted_queries": [
            str(item)
            for item in _as_list(raw.get("targeted_queries"))
            if str(item).strip()
        ][:2],
        "questions_for_clinician": [
            str(item)
            for item in _as_list(raw.get("questions_for_clinician"))
            if str(item).strip()
        ],
        "safety_flags": [
            str(item)
            for item in _as_list(raw.get("safety_flags"))
            if str(item).strip()
        ],
        "summary": str(raw.get("summary", "")),
        "model_fallback": "_fallback_reason" in raw,
    }
    return {
        "critique_result": critique,
        "traceability_issues": deterministic_issues,
        "requires_clinician_review": (
            state.get("requires_clinician_review", False)
            or verdict != "pass"
            or bool(critique["questions_for_clinician"])
        ),
        "audit_events": [
            _event(
                "EvidenceSafetyCritic",
                "independent_review_completed",
                verdict=verdict,
                issue_count=len(issues),
                targeted_queries=len(critique["targeted_queries"]),
            )
        ],
    }


def route_after_critic(
    state: MedicalState,
) -> Literal["targeted_review", "finalize", "safe_finalize"]:
    critique = state.get("critique_result", {})
    can_retry = state.get("revision_count", 0) < state.get(
        "max_revision_rounds", MAX_REFLECTION_ROUNDS
    )
    if (
        critique.get("verdict") == "revise"
        and can_retry
        and _as_list(critique.get("targeted_queries"))
    ):
        return "targeted_review"
    if critique.get("verdict") == "pass":
        return "finalize"
    return "safe_finalize"


async def targeted_review_node(state: MedicalState) -> Dict[str, Any]:
    queries = [
        str(item)
        for item in _as_list(
            state.get("critique_result", {}).get("targeted_queries")
        )
    ][:2]
    round_number = state.get("revision_count", 0) + 1
    evidence, events = await _retrieve_knowledge(
        queries, f"R{round_number}-", "TargetedReviewAgent"
    )
    events.append(
        _event(
            "TargetedReviewAgent",
            "bounded_reflection_round_completed",
            round=round_number,
            evidence_added=len(evidence),
        )
    )
    return {
        "evidence": evidence,
        "revision_count": round_number,
        "audit_events": events,
    }


def _escape_markdown(value: Any) -> str:
    return str(value or "").replace("|", "\\|").strip()


def _format_refs(ids: List[str], evidence_index: Dict[str, EvidenceItem]) -> str:
    parts = []
    for evidence_id in ids:
        item = evidence_index.get(evidence_id)
        if item:
            parts.append(f"[{evidence_id}] {_escape_markdown(item.get('statement'))}")
        else:
            parts.append(f"[{evidence_id}]（无对应证据）")
    return "；".join(parts) if parts else "未绑定可追溯证据"


async def finalize_node(state: MedicalState) -> Dict[str, Any]:
    synthesis = state.get("synthesis_result", {})
    critique = state.get("critique_result", {})
    triage = state.get("triage_result", {})
    evidence = [item for item in state.get("evidence", []) if isinstance(item, dict)]
    evidence_index = {
        item.get("evidence_id"): item for item in evidence if item.get("evidence_id")
    }

    lines = [
        "## 临床决策支持报告",
        "",
        "> 本报告面向医疗专业人员，是待复核的决策支持草案；不替代面诊、急救判断、处方或医嘱。",
        "",
        "### 分诊与病例摘要",
        "",
        f"- 紧急程度：**{_escape_markdown(triage.get('urgency', '未知'))}**",
        f"- 分诊依据：{_escape_markdown(triage.get('rationale', '未提供'))}",
        f"- 临床摘要：{_escape_markdown(synthesis.get('clinical_summary', '未形成'))}",
    ]

    if critique.get("review_mode") == "demo_presentation":
        lines.extend(["", "> 模拟数据演示报告：允许展示候选诊断，不代表真实临床验证或可直接执行的医嘱。",
                      "> 当前采用演示审校策略；真实患者使用前需切换标准策略并重新验证。"])

    if synthesis.get("abstain"):
        lines.extend(
            [
                "",
                "### 系统拒答 / 暂不收敛",
                "",
                _escape_markdown(
                    synthesis.get("abstain_reason", "当前信息不足，不能可靠排序。")
                ),
            ]
        )
    else:
        lines.extend(["", "### 排序鉴别诊断", ""])
        for item in _as_list(synthesis.get("ranked_differential")):
            if not isinstance(item, dict):
                continue
            lines.extend(
                [
                    f"**{item.get('rank', '-')}. {_escape_markdown(item.get('diagnosis'))}** "
                    f"（把握度：{_escape_markdown(item.get('confidence', 'low'))}；"
                    f"时效：{_escape_markdown(item.get('urgency', 'routine'))}）",
                    f"- 临床理由：{_escape_markdown(item.get('evidence_rationale'))}",
                    "- 支持证据："
                    + _format_refs(
                        _as_list(item.get("supporting_evidence_ids")), evidence_index
                    ),
                    "- 反证/不一致："
                    + _format_refs(
                        _as_list(item.get("contradicting_evidence_ids")), evidence_index
                    ),
                    f"- 最有区分度的下一步：{_escape_markdown(item.get('next_step'))}",
                    "",
                ]
            )

    actions = _as_list(synthesis.get("recommended_actions"))
    if actions:
        lines.extend(["### 建议的下一步（由临床医生决定）", ""])
        for item in actions:
            if not isinstance(item, dict):
                continue
            refs = _format_refs(_as_list(item.get("evidence_ids")), evidence_index)
            lines.append(
                f"- **{_escape_markdown(item.get('priority', 'routine'))}**："
                f"{_escape_markdown(item.get('action'))} — "
                f"{_escape_markdown(item.get('purpose'))}（依据：{refs}）"
            )

    questions = list(_as_list(synthesis.get("unresolved_questions")))
    questions.extend(
        item
        for item in _as_list(critique.get("questions_for_clinician"))
        if item not in questions
    )
    if questions:
        lines.extend(["", "### 仍需补充或确认", ""])
        lines.extend(f"- {_escape_markdown(item)}" for item in questions)

    lines.extend(["", "### 独立审校", ""])
    lines.append(
        f"- 结论：**{_escape_markdown(critique.get('verdict', '未完成'))}**"
    )
    if critique.get("summary"):
        lines.append(f"- 摘要：{_escape_markdown(critique.get('summary'))}")
    for warning in _as_list(critique.get("warnings")):
        lines.append(f"- 演示提示（不阻止展示）：{_escape_markdown(warning.get('description', ''))}")
    for issue in _as_list(critique.get("issues"))[:6]:
        if isinstance(issue, dict):
            lines.append(
                f"- [{_escape_markdown(issue.get('severity', 'unknown'))}] "
                f"{_escape_markdown(issue.get('description'))}"
            )

    lines.extend(["", "### 证据索引", ""])
    if not evidence:
        lines.append("- 无可追溯证据；必须由临床医生补充资料后再评估。")
    for item in evidence:
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        locator = _source_locator(source) or "无定位信息"
        demo_label = "；演示数据" if source.get("is_demo") else ""
        lines.append(
            f"- **[{item.get('evidence_id')}]** {_escape_markdown(item.get('statement'))} "
            f"— {_escape_markdown(source.get('name', '未知来源'))} "
            f"({_escape_markdown(locator)}{demo_label})"
        )

    requires_review = True
    if state.get("knowledge_hit"):
        lines.extend(["", "**本次草案等待医生批准；历史审校与点赞不代表本次已获批准。**"])
    elif state.get("requires_clinician_review") or critique.get("verdict") != "pass":
        lines.extend(
            [
                "",
                "**复核优先级：高。存在急迫性、信息缺口或审校未完全通过，禁止直接转化为自动医嘱。**",
            ]
        )
    report = "\n".join(lines).strip()
    return {
        "final_report": report,
        "requires_clinician_review": requires_review,
        "messages": [AIMessage(content=report, name="ClinicalHost")],
        "audit_events": [
            _event(
                "ClinicalHost",
                "clinician_facing_report_finalized",
                evidence_count=len(evidence),
                requires_clinician_review=requires_review,
            )
        ],
    }


async def safe_finalize_node(state: MedicalState) -> Dict[str, Any]:
    """Fail closed when independent review does not pass."""

    triage = state.get("triage_result", {})
    critique = state.get("critique_result", {})
    issues = [
        item for item in _as_list(critique.get("issues")) if isinstance(item, dict)
    ][:6]
    questions = [
        str(item)
        for item in _as_list(critique.get("questions_for_clinician"))
        if str(item).strip()
    ]
    lines = [
        "## 临床决策支持安全升级",
        "",
        "> 独立审校未通过，本系统不展示或执行未经审校的诊断、检查、转诊或治疗建议。",
        "",
        f"- 分诊级别：**{_escape_markdown(triage.get('urgency', '未知'))}**",
        f"- 审校结论：**{_escape_markdown(critique.get('verdict', 'escalate'))}**",
        f"- 审校摘要：{_escape_markdown(critique.get('summary', '需要临床人员复核'))}",
    ]
    if issues:
        lines.extend(["", "### 需要处理的问题", ""])
        for item in issues:
            lines.append(
                f"- [{_escape_markdown(item.get('severity', 'unknown'))}] "
                f"{_escape_markdown(item.get('description', '未说明'))}"
            )
    if questions:
        lines.extend(["", "### 需要医生补充或确认", ""])
        lines.extend(f"- {_escape_markdown(item)}" for item in questions)
    lines.extend(
        [
            "",
            "**请由具备资质的临床人员复核原始资料；本次流程不会创建医学检查申请。**",
        ]
    )
    report = "\n".join(lines).strip()
    return {
        "final_report": report,
        "requires_clinician_review": True,
        "messages": [AIMessage(content=report, name="EvidenceSafetyCritic")],
        "audit_events": [
            _event(
                "EvidenceSafetyCritic",
                "unsafe_output_suppressed_and_escalated",
                verdict=critique.get("verdict", "escalate"),
                issue_count=len(issues),
            )
        ],
    }


def _eligible_test_actions(state: MedicalState) -> List[Dict[str, Any]]:
    valid_evidence_ids = {
        item.get("evidence_id")
        for item in state.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    eligible: List[Dict[str, Any]] = []
    for item in _as_list(state.get("synthesis_result", {}).get("recommended_actions")):
        if not isinstance(item, dict) or item.get("type") != "test":
            continue
        evidence_ids = [
            str(value) for value in _as_list(item.get("evidence_ids"))
        ]
        if not evidence_ids or any(value not in valid_evidence_ids for value in evidence_ids):
            continue
        eligible.append(
            {
                "test": str(item.get("action", "")).strip(),
                "indication": str(item.get("purpose", "")).strip(),
                "priority": str(item.get("priority", "routine")),
                "evidence_ids": evidence_ids,
            }
        )
    return [item for item in eligible if item["test"]]


def route_after_finalize(state: MedicalState) -> Literal["prepare_exam_order", "end"]:
    if state.get("critique_result", {}).get("verdict") != "pass":
        return "end"
    return "prepare_exam_order" if state.get("knowledge_hit") or _eligible_test_actions(state) else "end"


async def prepare_exam_order_node(state: MedicalState) -> Dict[str, Any]:
    """Build an immutable proposal; this node never performs a write action."""

    items = _eligible_test_actions(state)
    if not items and not state.get("knowledge_hit"):
        return {
            "exam_order_proposal": {},
            "audit_events": [
                _event("ExamOrderProposalBuilder", "no_eligible_test_actions")
            ],
        }

    created = datetime.now(timezone.utc)
    expires = created + timedelta(seconds=EXAM_ORDER_APPROVAL_TTL_SECONDS)
    seed = json.dumps(
        {
            "case_id": state.get("case_id"),
            "patient_id": state.get("patient_id"),
            "items": items,
            "created_at": created.isoformat(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    proposal_id = "EXAM-PROP-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    proposal: Dict[str, Any] = {
        "proposal_id": proposal_id,
        "case_id": state.get("case_id", ""),
        "patient_id": state.get("patient_id", ""),
        "version": 1,
        "items": items,
        "safety_flags": [
            str(item)
            for item in _as_list(state.get("critique_result", {}).get("safety_flags"))
        ],
        "created_at": created.isoformat(),
        "expires_at": expires.isoformat(),
        "status": "pending_approval",
    }
    if state.get("knowledge_hit"):
        proposal.update({
            "kind": "reference_draft_review",
            "draft_hash": hashlib.sha256(state.get("final_report", "").encode()).hexdigest(),
            "knowledge_entry_id": state["knowledge_hit"]["entry_id"],
        })
    proposal["payload_hash"] = compute_exam_order_payload_hash(proposal)
    proposal["idempotency_key"] = (
        f"exam-order:{proposal_id}:v{proposal['version']}:{proposal['payload_hash']}"
    )
    return {
        "exam_order_proposal": proposal,
        "audit_events": [
            _event(
                "ExamOrderProposalBuilder",
                "immutable_exam_order_proposal_created",
                proposal_id=proposal_id,
                version=proposal["version"],
                payload_hash=proposal["payload_hash"],
                item_count=len(items),
            )
        ],
    }


def route_after_proposal(state: MedicalState) -> Literal["await_approval", "end"]:
    proposal = state.get("exam_order_proposal", {})
    return "await_approval" if proposal.get("status") == "pending_approval" else "end"


def _proposal_is_expired(proposal: Dict[str, Any]) -> bool:
    try:
        expires_at = datetime.fromisoformat(
            str(proposal.get("expires_at", "")).replace("Z", "+00:00")
        )
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at.astimezone(timezone.utc) <= datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return True


async def await_clinician_approval_node(state: MedicalState) -> Dict[str, Any]:
    proposal = state.get("exam_order_proposal", {})
    resumed = interrupt(
        {
            "tool": "review_reference_draft" if proposal.get("kind") == "reference_draft_review" else "create_exam_order",
            "args": {"proposal": proposal},
            "message": ("本次直接引用旧方案，未重新运行 Critic；请医生核对当前患者适用性并批准或拒绝。"
                        if proposal.get("kind") == "reference_draft_review"
                        else "医学检查申请属于写操作，请医生核对准确草案后批准或拒绝。"),
        }
    )
    raw = resumed if isinstance(resumed, dict) else {}
    decision = str(raw.get("decision", "invalid"))
    binding_fields = (
        "proposal_id",
        "case_id",
        "patient_id",
        "version",
        "payload_hash",
    )
    binding_valid = all(raw.get(field) == proposal.get(field) for field in binding_fields)
    payload_valid = proposal.get("payload_hash") == compute_exam_order_payload_hash(proposal)
    if proposal.get("kind") == "reference_draft_review":
        payload_valid = payload_valid and proposal.get("draft_hash") == hashlib.sha256(
            state.get("final_report", "").encode()
        ).hexdigest()
    identity_valid = bool(str(raw.get("approver_id", "")).strip()) and bool(
        str(raw.get("approver_role", "")).strip()
    )
    if _proposal_is_expired(proposal):
        decision = "expired"
    elif decision not in {"approved", "rejected"}:
        decision = "invalid"
    elif not binding_valid or not payload_valid or not identity_valid:
        decision = "invalid"

    approval = {
        "decision_id": str(raw.get("decision_id", "")),
        "proposal_id": proposal.get("proposal_id", ""),
        "case_id": proposal.get("case_id", ""),
        "patient_id": proposal.get("patient_id", ""),
        "version": proposal.get("version", 0),
        "payload_hash": proposal.get("payload_hash", ""),
        "decision": decision,
        "approver_id": str(raw.get("approver_id", "")),
        "approver_role": str(raw.get("approver_role", "")),
        "reason": str(raw.get("reason", "")),
        "decided_at": str(raw.get("decided_at", _now())),
    }
    return {
        "approval_decision": approval,
        "audit_events": [
            _event(
                "ClinicianApprovalGate",
                "clinician_decision_recorded",
                proposal_id=proposal.get("proposal_id"),
                version=proposal.get("version"),
                decision=decision,
                approver_id=approval["approver_id"],
            )
        ],
    }


def route_after_approval(state: MedicalState) -> Literal["execute", "stop"]:
    return (
        "execute"
        if state.get("approval_decision", {}).get("decision") == "approved"
        else "stop"
    )


async def execute_exam_order_node(state: MedicalState) -> Dict[str, Any]:
    proposal = state.get("exam_order_proposal", {})
    approval = state.get("approval_decision", {})
    validation_error = validate_exam_order_approval(proposal, approval)
    if validation_error:
        result = {
            "status": "execution_blocked",
            "proposal_id": proposal.get("proposal_id", ""),
            "order_id": "",
            "idempotency_key": proposal.get("idempotency_key", ""),
            "executed_at": _now(),
            "is_demo": True,
            "message": validation_error,
        }
    elif proposal.get("kind") == "reference_draft_review" and not proposal.get("items"):
        result = {
            "status": "review_approved", "proposal_id": proposal.get("proposal_id"),
            "order_id": "", "idempotency_key": proposal.get("idempotency_key"),
            "executed_at": _now(), "is_demo": True,
            "message": "医生已批准本次引用方案；没有检查项目，未创建检查申请。",
        }
    else:
        try:
            result = await create_exam_order.ainvoke(
                {"proposal": proposal, "approval": approval}
            )
        except Exception as exc:
            result = {
                "status": "execution_failed",
                "proposal_id": proposal.get("proposal_id", ""),
                "order_id": "",
                "idempotency_key": proposal.get("idempotency_key", ""),
                "executed_at": _now(),
                "is_demo": True,
                "message": f"检查申请执行失败并已安全停止：{exc}",
            }
    updated_proposal = {**proposal, "status": result.get("status", "execution_failed")}
    message = (
        "## 医学检查 HITL 执行结果\n\n"
        f"- 状态：**{_escape_markdown(result.get('status'))}**\n"
        f"- Proposal：`{_escape_markdown(result.get('proposal_id'))}`\n"
        f"- Order：`{_escape_markdown(result.get('order_id') or '未创建')}`\n"
        f"- 说明：{_escape_markdown(result.get('message'))}"
    )
    return {
        "exam_order_proposal": updated_proposal,
        "exam_order_execution_result": result,
        "messages": [AIMessage(content=message, name="ExamOrderExecutor")],
        "audit_events": [
            _event(
                "ExamOrderExecutor",
                "exam_order_execution_finished",
                proposal_id=proposal.get("proposal_id"),
                status=result.get("status"),
                order_id=result.get("order_id"),
            )
        ],
    }


async def stop_without_execution_node(state: MedicalState) -> Dict[str, Any]:
    proposal = state.get("exam_order_proposal", {})
    approval = state.get("approval_decision", {})
    decision = str(approval.get("decision", "invalid"))
    reason = str(approval.get("reason", "")).strip() or "未获得有效医生批准。"
    result = {
        "status": decision,
        "proposal_id": proposal.get("proposal_id", ""),
        "order_id": "",
        "idempotency_key": proposal.get("idempotency_key", ""),
        "executed_at": _now(),
        "is_demo": True,
        "message": reason,
    }
    updated_proposal = {**proposal, "status": decision}
    message = (
        "## 医学检查 HITL 结果\n\n"
        f"- 状态：**{_escape_markdown(decision)}**\n"
        "- 未创建任何检查申请。\n"
        f"- 原因：{_escape_markdown(reason)}"
    )
    return {
        "exam_order_proposal": updated_proposal,
        "exam_order_execution_result": result,
        "messages": [AIMessage(content=message, name="ClinicianApprovalGate")],
        "audit_events": [
            _event(
                "ClinicianApprovalGate",
                "exam_order_stopped_without_execution",
                proposal_id=proposal.get("proposal_id"),
                decision=decision,
            )
        ],
    }


def route_at_start(state: MedicalState) -> Literal["reuse", "standard"]:
    safety_text = "\n".join([state.get("chief_complaint", ""),
                             *_present_context_text(state.get("clinical_context", {}))])
    return ("reuse" if state.get("retrieved_plans") and not _hard_safety_check(safety_text)["red_flags"]
            else "standard")


async def reuse_liked_plan_node(state: MedicalState) -> Dict[str, Any]:
    """Quote the full historical plan, preserving new identity and approval."""
    hit = state["retrieved_plans"][0]
    plan = validate_plan(hit["plan"])
    saved_at = datetime.fromisoformat(hit["saved_at"]).date().isoformat()
    report = "\n".join([
        "## 历史诊疗方案引用 · 待本次医生审批",
        f"- 当前患者：{_escape_markdown(state['patient_id'])}",
        f"- 当前主诉：{_escape_markdown(state['chief_complaint'])}",
        "- 当前结构化病情（未经本次模型分析）：",
        _escape_markdown(json.dumps(state.get("clinical_context", {}), ensure_ascii=False)),
        "",
        f"- 主诉文字匹配分数：{hit['score']:.2f}；收录日期：{saved_at}。",
        "**本次未重新运行分诊模型、诊疗分析或 Critic。主诉相似不代表病情相同。**",
        "**请医生核对年龄、过敏、用药、妊娠、生命体征及当前禁忌；批准前不能执行。**",
        "",
        "### 引用的历史完整方案",
        "以下分诊、患者事实、检查结果与审校结论均属于历史方案，不是本次患者已确认的事实或审批。",
        "",
        *["> " + line for line in plan["report_text"].splitlines()],
        "",
        "**本次审批状态：待批准；历史点赞和历史 Critic 通过均不代表本次已批准。**",
    ])
    return {
        "synthesis_result": plan["synthesis_result"],
        "evidence": plan["evidence"],
        "critique_result": {"verdict": "not_run", "summary": "本次跳过 Critic，仅由医生审核引用方案。"},
        "knowledge_hit": {"entry_id": hit["entry_id"], "mode": "direct_plan", "saved_at": saved_at},
        "final_report": report,
        "requires_clinician_review": True,
        "messages": [AIMessage(content=report, name="LikedPlanRetrieval")],
        "audit_events": [_event("AnswerKnowledgeBase", "liked_plan_quoted_pending_human_review",
                                entry_id=hit["entry_id"], score=hit["score"])],
    }


# --------------------------- Graph topology ---------------------------
workflow = StateGraph(MedicalState)
workflow.add_node("intake_triage", intake_triage_node)
workflow.add_node("emergency_response", emergency_response_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("patient_data", patient_data_node)
workflow.add_node("knowledge_evidence", knowledge_evidence_node)
workflow.add_node("specialist_panel", specialist_panel_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("critic", critic_node)
workflow.add_node("targeted_review", targeted_review_node)
workflow.add_node("finalize", finalize_node)
workflow.add_node("safe_finalize", safe_finalize_node)
workflow.add_node("prepare_exam_order", prepare_exam_order_node)
workflow.add_node("await_clinician_approval", await_clinician_approval_node)
workflow.add_node("execute_exam_order", execute_exam_order_node)
workflow.add_node("stop_without_execution", stop_without_execution_node)

workflow.add_node("reuse_liked_plan", reuse_liked_plan_node)
workflow.add_conditional_edges(START, route_at_start, {"reuse": "reuse_liked_plan", "standard": "intake_triage"})
workflow.add_edge("reuse_liked_plan", "prepare_exam_order")
workflow.add_conditional_edges(
    "intake_triage",
    route_after_triage,
    {"emergency": "emergency_response", "standard": "supervisor"},
)
workflow.add_edge("emergency_response", END)

# Three complementary lanes run in parallel after the central host plans work.
workflow.add_edge("supervisor", "patient_data")
workflow.add_edge("supervisor", "knowledge_evidence")
workflow.add_edge("supervisor", "specialist_panel")
workflow.add_edge(
    ["patient_data", "knowledge_evidence", "specialist_panel"], "synthesis"
)

workflow.add_edge("synthesis", "critic")
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "targeted_review": "targeted_review",
        "finalize": "finalize",
        "safe_finalize": "safe_finalize",
    },
)
workflow.add_edge("targeted_review", "synthesis")
workflow.add_edge("safe_finalize", END)
workflow.add_conditional_edges(
    "finalize",
    route_after_finalize,
    {"prepare_exam_order": "prepare_exam_order", "end": END},
)
workflow.add_conditional_edges(
    "prepare_exam_order",
    route_after_proposal,
    {"await_approval": "await_clinician_approval", "end": END},
)
workflow.add_conditional_edges(
    "await_clinician_approval",
    route_after_approval,
    {"execute": "execute_exam_order", "stop": "stop_without_execution"},
)
workflow.add_edge("execute_exam_order", END)
workflow.add_edge("stop_without_execution", END)

medical_graph = workflow.compile(checkpointer=memory)
