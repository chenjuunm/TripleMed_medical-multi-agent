"""Shared state contracts for the clinical decision-support graph.

The graph deliberately keeps clinical facts, evidence, hypotheses and chat
messages in separate channels. Treating the message history as the only form
of memory makes provenance and parallel agent execution unnecessarily fragile.
"""

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PatientSnapshot(TypedDict, total=False):
    """Normalized case facts. Unknown is different from explicitly absent."""

    demographics: Dict[str, Any]
    chief_complaint: str
    present_findings: List[Dict[str, Any]]
    explicit_absent_findings: List[Dict[str, Any]]
    time_course: str
    medications: List[str]
    allergies: List[str]
    history: List[str]
    available_tests: List[Dict[str, Any]]
    unknown_critical_fields: List[str]


class EvidenceItem(TypedDict, total=False):
    """One auditable fact returned by a patient-data or knowledge tool."""

    evidence_id: str
    kind: str
    statement: str
    source: Dict[str, Any]
    query: Dict[str, Any]
    purpose: str
    reliability: str
    agent: str
    retrieved_at: str


class SpecialistOpinion(TypedDict, total=False):
    """An independent diagnostic viewpoint; it is not evidence by itself."""

    opinion_id: str
    role: str
    focus: str
    candidate_hypotheses: List[Dict[str, Any]]
    missing_information: List[str]
    certainty: str


class ExamOrderProposal(TypedDict, total=False):
    """Immutable, non-executable examination-order proposal awaiting approval."""

    proposal_id: str
    case_id: str
    patient_id: str
    version: int
    items: List[Dict[str, Any]]
    safety_flags: List[str]
    payload_hash: str
    idempotency_key: str
    created_at: str
    expires_at: str
    status: str


class ClinicianApprovalDecision(TypedDict, total=False):
    """Trusted decision bound to one exact proposal payload."""

    decision_id: str
    proposal_id: str
    case_id: str
    patient_id: str
    version: int
    payload_hash: str
    decision: str
    approver_id: str
    approver_role: str
    reason: str
    decided_at: str


class ExamOrderExecutionResult(TypedDict, total=False):
    """Result of the idempotent examination-order action boundary."""

    status: str
    proposal_id: str
    order_id: str
    idempotency_key: str
    executed_at: str
    is_demo: bool
    replayed: bool
    message: str


class MedicalState(TypedDict, total=False):
    # Encounter identity and input
    case_id: str
    patient_id: str
    chief_complaint: str
    clinical_context: Dict[str, Any]
    retrieved_plans: List[Dict[str, Any]]
    knowledge_hit: Dict[str, Any]

    # Conversation output is kept for LangGraph/FastAPI streaming only.
    messages: Annotated[List[BaseMessage], add_messages]

    # Structured clinical working memory
    case_snapshot: PatientSnapshot
    triage_result: Dict[str, Any]
    work_plan: Dict[str, Any]

    # Parallel workers append to these reducer-backed ledgers.
    evidence: Annotated[List[EvidenceItem], operator.add]
    specialist_opinions: Annotated[List[SpecialistOpinion], operator.add]
    audit_events: Annotated[List[Dict[str, Any]], operator.add]

    # Host synthesis and independent review
    synthesis_result: Dict[str, Any]
    critique_result: Dict[str, Any]
    traceability_issues: List[Dict[str, Any]]
    final_report: str
    requires_clinician_review: bool

    # Human-in-the-loop boundary for examination-order write actions.
    exam_order_proposal: ExamOrderProposal
    approval_decision: ClinicianApprovalDecision
    exam_order_execution_result: ExamOrderExecutionResult

    # Bounded reflection loop
    revision_count: int
    max_revision_rounds: int

    # Backwards-compatible fields used by the existing API/UI.
    plan_result: Dict[str, Any]
    reasoning_result: Dict[str, Any]
    pending_tool_call: Dict[str, Any]
