"""Tests use fictional cases, identities, contacts, and credentials only.

Values that resemble personal or medical data are synthetic test fixtures and
do not represent real people or records.
"""

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

import main
from agents import IntakeTriageOutput, SynthesisOutput, TriageOutput
from graph import (
    _hard_safety_check,
    _present_context_text,
    await_clinician_approval_node,
    emergency_response_node,
    execute_exam_order_node,
    medical_graph,
    prepare_exam_order_node,
    route_after_approval,
    route_after_critic,
    safe_finalize_node,
    stop_without_execution_node,
    validate_traceability,
)
from rag_system import _tokenize_for_bm25
from state import MedicalState
from tools import (
    _DEMO_ORDER_STORE,
    compute_exam_order_payload_hash,
    create_exam_order,
)


def _build_hitl_test_graph():
    workflow = StateGraph(MedicalState)
    workflow.add_node("prepare", prepare_exam_order_node)
    workflow.add_node("approval", await_clinician_approval_node)
    workflow.add_node("execute", execute_exam_order_node)
    workflow.add_node("stop", stop_without_execution_node)
    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "approval")
    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {"execute": "execute", "stop": "stop"},
    )
    workflow.add_edge("execute", END)
    workflow.add_edge("stop", END)
    return workflow.compile(checkpointer=MemorySaver())


async def _prime_hitl_test_graph(graph, config):
    async for _ in graph.astream(
        HitlWorkflowTests._base_state(), config, stream_mode="updates"
    ):
        pass
    return await graph.aget_state(config)


class SafetyRuleTests(unittest.TestCase):
    def test_negated_emergency_phrases_do_not_trigger(self):
        result = _hard_safety_check(
            "患者否认突发剧烈胸痛，无严重呼吸困难"
        )
        self.assertEqual(result, {"urgency": "low", "red_flags": []})

    def test_coordinated_negation_does_not_trigger(self):
        for text in (
            "患者否认胸痛及严重呼吸困难",
            "患者无胸痛、严重呼吸困难",
        ):
            with self.subTest(text=text):
                self.assertEqual(_hard_safety_check(text)["urgency"], "low")

    def test_affirmed_emergency_phrases_still_trigger(self):
        result = _hard_safety_check("突发剧烈胸痛并严重呼吸困难")
        self.assertEqual(result["urgency"], "emergency")
        self.assertIn("突发剧烈胸痛", result["red_flags"])

    def test_hypertension_without_end_organ_symptom_is_high(self):
        result = _hard_safety_check("血压 190/120，无胸痛、无呼吸困难")
        self.assertEqual(result["urgency"], "high")

    def test_hypertension_with_end_organ_symptom_is_emergency(self):
        result = _hard_safety_check("血压 190/120，伴剧烈头痛")
        self.assertEqual(result["urgency"], "emergency")

    def test_absent_context_is_excluded_from_safety_text(self):
        flattened = _present_context_text(
            {
                "explicit_absent_findings": ["胸痛", "呼吸困难"],
                "findings": [
                    {"status": "present", "name": "发热"},
                    {"status": "absent", "name": "气促"},
                ],
            }
        )
        self.assertEqual(flattened, ["发热"])


class StructuredOutputTests(unittest.TestCase):
    def test_compact_snapshot_lists_are_normalized(self):
        parsed = IntakeTriageOutput.model_validate(
            {
                "snapshot": {
                    "present_findings": ["活动后胸闷"],
                    "explicit_absent_findings": ["晕厥"],
                    "available_tests": ["心电图"],
                }
            }
        )
        self.assertEqual(parsed.snapshot.present_findings[0]["finding"], "活动后胸闷")
        self.assertEqual(
            parsed.snapshot.explicit_absent_findings[0]["finding"], "晕厥"
        )
        self.assertEqual(parsed.snapshot.available_tests[0]["test"], "心电图")

    def test_boolean_string_is_normalized(self):
        self.assertFalse(SynthesisOutput.model_validate({"abstain": "false"}).abstain)

    def test_invalid_red_flag_shape_is_rejected(self):
        with self.assertRaises(ValidationError):
            TriageOutput.model_validate({"red_flags": {"unexpected": "mapping"}})

    def test_chinese_bm25_tokenization_has_characters_and_bigrams(self):
        tokens = _tokenize_for_bm25("胸痛 ECG")
        self.assertIn("胸", tokens)
        self.assertIn("胸痛", tokens)
        self.assertIn("ecg", tokens)


class P0SafetyPathTests(unittest.IsolatedAsyncioTestCase):
    def test_active_graph_contains_hitl_and_safe_finalize_nodes(self):
        graph = medical_graph.get_graph()
        node_names = set(graph.nodes)
        self.assertTrue(
            {
                "safe_finalize",
                "prepare_exam_order",
                "await_clinician_approval",
                "execute_exam_order",
                "stop_without_execution",
            }.issubset(node_names)
        )

    async def test_emergency_path_suppresses_model_generated_actions(self):
        result = await emergency_response_node(
            {
                "triage_result": {
                    "red_flags": ["突发剧烈胸痛"],
                    "immediate_actions": ["立即给予阿司匹林"],
                }
            }
        )
        self.assertNotIn("阿司匹林", result["final_report"])
        self.assertIn("当地急救服务", result["final_report"])
        self.assertEqual(
            result["audit_events"][0]["details"]["model_actions_suppressed"], 1
        )

    def test_nonpassing_critic_routes_to_safe_finalize(self):
        self.assertEqual(
            route_after_critic(
                {
                    "critique_result": {"verdict": "revise", "targeted_queries": []},
                    "revision_count": 0,
                    "max_revision_rounds": 1,
                }
            ),
            "safe_finalize",
        )
        self.assertEqual(
            route_after_critic({"critique_result": {"verdict": "escalate"}}),
            "safe_finalize",
        )
        self.assertEqual(
            route_after_critic({"critique_result": {"verdict": "pass"}}),
            "finalize",
        )

    async def test_safe_finalize_hides_unreviewed_actions(self):
        result = await safe_finalize_node(
            {
                "triage_result": {"urgency": "medium"},
                "synthesis_result": {
                    "recommended_actions": [
                        {"type": "treatment_consideration", "action": "给予阿司匹林"}
                    ]
                },
                "critique_result": {
                    "verdict": "revise",
                    "summary": "证据不足",
                    "issues": [
                        {
                            "severity": "high",
                            "description": "治疗建议缺少可追溯依据",
                        }
                    ],
                },
            }
        )
        self.assertNotIn("给予阿司匹林", result["final_report"])
        self.assertIn("不展示或执行", result["final_report"])
        self.assertTrue(result["requires_clinician_review"])

    def test_test_recommendation_without_evidence_is_a_high_issue(self):
        issues = validate_traceability(
            {
                "evidence": [],
                "synthesis_result": {
                    "ranked_differential": [],
                    "recommended_actions": [
                        {"type": "test", "action": "心电图", "evidence_ids": []}
                    ],
                },
            }
        )
        self.assertTrue(
            any(
                issue["type"] == "unsupported_clinical_action"
                and issue["severity"] == "high"
                for issue in issues
            )
        )


class HitlWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _DEMO_ORDER_STORE.clear()

    @staticmethod
    def _base_state():
        return {
            "case_id": "CASE-HITL-001",
            "patient_id": "P001",
            "evidence": [
                {
                    "evidence_id": "K001",
                    "kind": "medical_knowledge",
                    "statement": "胸痛应进行心电图评估。",
                    "source": {"source_id": "GUIDE-001"},
                }
            ],
            "synthesis_result": {
                "recommended_actions": [
                    {
                        "action": "心电图",
                        "purpose": "评估心肌缺血",
                        "priority": "now",
                        "type": "test",
                        "evidence_ids": ["K001"],
                    }
                ]
            },
            "critique_result": {"verdict": "pass", "safety_flags": []},
            "messages": [],
            "audit_events": [],
        }

    @staticmethod
    def _approval_for(proposal, decision="approved"):
        return {
            "decision_id": "DECISION-001",
            "proposal_id": proposal["proposal_id"],
            "case_id": proposal["case_id"],
            "patient_id": proposal["patient_id"],
            "version": proposal["version"],
            "payload_hash": proposal["payload_hash"],
            "decision": decision,
            "approver_id": "doctor-001",
            "approver_role": "clinician",
            "reason": "" if decision == "approved" else "需要补充资料",
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }

    async def test_proposal_is_versioned_hashed_and_non_executable(self):
        result = await prepare_exam_order_node(self._base_state())
        proposal = result["exam_order_proposal"]
        self.assertEqual(proposal["status"], "pending_approval")
        self.assertEqual(proposal["version"], 1)
        self.assertEqual(
            proposal["payload_hash"], compute_exam_order_payload_hash(proposal)
        )
        self.assertTrue(proposal["idempotency_key"].startswith("exam-order:"))

    async def test_action_rejects_tampered_proposal(self):
        proposal = (await prepare_exam_order_node(self._base_state()))[
            "exam_order_proposal"
        ]
        approval = self._approval_for(proposal)
        proposal["items"][0]["test"] = "已被篡改的检查"
        with self.assertRaises(ValueError):
            await create_exam_order.ainvoke(
                {"proposal": proposal, "approval": approval}
            )

    async def test_action_is_idempotent(self):
        proposal = (await prepare_exam_order_node(self._base_state()))[
            "exam_order_proposal"
        ]
        approval = self._approval_for(proposal)
        first = await create_exam_order.ainvoke(
            {"proposal": proposal, "approval": approval}
        )
        second = await create_exam_order.ainvoke(
            {"proposal": proposal, "approval": approval}
        )
        self.assertEqual(first["order_id"], second["order_id"])
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])

    async def test_action_rejects_tampered_idempotency_key(self):
        proposal = (await prepare_exam_order_node(self._base_state()))[
            "exam_order_proposal"
        ]
        approval = self._approval_for(proposal)
        proposal["idempotency_key"] = "attacker-controlled-key"
        with self.assertRaises(ValueError):
            await create_exam_order.ainvoke(
                {"proposal": proposal, "approval": approval}
            )

    async def test_langgraph_interrupt_resume_executes_exact_approved_proposal(self):
        graph = _build_hitl_test_graph()
        config = {"configurable": {"thread_id": "hitl-integration-test"}}

        first_events = []
        async for event in graph.astream(
            self._base_state(), config, stream_mode="updates"
        ):
            first_events.append(event)
        self.assertTrue(any("__interrupt__" in event for event in first_events))

        snapshot = await graph.aget_state(config)
        proposal = snapshot.values["exam_order_proposal"]
        approval = self._approval_for(proposal)
        async for _ in graph.astream(
            Command(resume=approval), config, stream_mode="updates"
        ):
            pass

        final_snapshot = await graph.aget_state(config)
        execution = final_snapshot.values["exam_order_execution_result"]
        self.assertEqual(execution["status"], "executed")
        self.assertTrue(execution["order_id"].startswith("DEMO-EXAM-"))

    async def test_langgraph_rejection_stops_without_execution(self):
        graph = _build_hitl_test_graph()
        config = {"configurable": {"thread_id": "hitl-rejection-test"}}
        async for _ in graph.astream(
            self._base_state(), config, stream_mode="updates"
        ):
            pass
        proposal = (await graph.aget_state(config)).values["exam_order_proposal"]
        rejection = self._approval_for(proposal, decision="rejected")
        async for _ in graph.astream(
            Command(resume=rejection), config, stream_mode="updates"
        ):
            pass
        final_values = (await graph.aget_state(config)).values
        self.assertEqual(final_values["exam_order_execution_result"]["status"], "rejected")
        self.assertEqual(final_values["exam_order_execution_result"]["order_id"], "")
        self.assertFalse(_DEMO_ORDER_STORE)

    async def test_expired_proposal_cannot_execute(self):
        proposal = (await prepare_exam_order_node(self._base_state()))[
            "exam_order_proposal"
        ]
        proposal["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        proposal["payload_hash"] = compute_exam_order_payload_hash(proposal)
        approval = self._approval_for(proposal)
        with self.assertRaises(ValueError):
            await create_exam_order.ainvoke(
                {"proposal": proposal, "approval": approval}
            )


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        main.case_threads.clear()
        main.case_patients.clear()
        main.approval_claims.clear()
        self.client = TestClient(main.app)

    def test_whitespace_only_input_is_rejected(self):
        response = self.client.post(
            "/medical_chat",
            json={"patient_id": "   ", "chief_complaint": "   "},
        )
        self.assertEqual(response.status_code, 422)

    def test_unavailable_lm_studio_fails_before_workflow(self):
        status = {
            "ready": False,
            "endpoint": "http://127.0.0.1:1234/v1/models",
            "loaded_models": [],
            "missing_models": ["medical-main-qwen38-27b"],
            "error": "offline",
        }
        with patch.object(main, "lm_studio_runtime_status", AsyncMock(return_value=status)):
            response = self.client.post(
                "/medical_chat",
                json={"patient_id": "P001", "chief_complaint": "咽痛三天"},
            )
        self.assertEqual(response.status_code, 503)

    def test_confirmation_requires_case_id(self):
        response = self.client.post(
            "/confirm_tool",
            json={
                "patient_id": "P001",
                "proposal_id": "PROP-1",
                "proposal_version": 1,
                "payload_hash": "a" * 64,
                "approved": True,
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_confirmation_rejects_patient_mismatch(self):
        main.case_threads["CASE-1"] = "thread-1"
        main.case_patients["CASE-1"] = "P001"
        with patch.object(main, "CLINICIAN_APPROVAL_TOKEN", "test-token"):
            response = self.client.post(
                "/confirm_tool",
                json={
                    "patient_id": "P002",
                    "case_id": "CASE-1",
                    "proposal_id": "PROP-1",
                    "proposal_version": 1,
                    "payload_hash": "a" * 64,
                    "approved": True,
                },
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Clinician-Id": "doctor-001",
                    "X-Clinician-Role": "clinician",
                },
            )
        self.assertEqual(response.status_code, 403)

    def test_confirmation_fails_closed_without_server_token(self):
        with patch.object(main, "CLINICIAN_APPROVAL_TOKEN", ""):
            response = self.client.post(
                "/confirm_tool",
                json={
                    "patient_id": "P001",
                    "case_id": "CASE-1",
                    "proposal_id": "PROP-1",
                    "proposal_version": 1,
                    "payload_hash": "a" * 64,
                    "approved": True,
                },
            )
        self.assertEqual(response.status_code, 503)

    def test_confirmation_rejects_unauthorized_role(self):
        with patch.object(main, "CLINICIAN_APPROVAL_TOKEN", "test-token"):
            response = self.client.post(
                "/confirm_tool",
                json={
                    "patient_id": "P001",
                    "case_id": "CASE-1",
                    "proposal_id": "PROP-1",
                    "proposal_version": 1,
                    "payload_hash": "a" * 64,
                    "approved": True,
                },
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Clinician-Id": "viewer-001",
                    "X-Clinician-Role": "viewer",
                },
            )
        self.assertEqual(response.status_code, 403)

    def test_rejection_requires_reason(self):
        response = self.client.post(
            "/confirm_tool",
            json={
                "patient_id": "P001",
                "case_id": "CASE-1",
                "proposal_id": "PROP-1",
                "proposal_version": 1,
                "payload_hash": "a" * 64,
                "approved": False,
                "reason": "",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_authenticated_api_resumes_exact_proposal(self):
        graph = _build_hitl_test_graph()
        config = {"configurable": {"thread_id": "api-hitl-thread"}}
        proposal = asyncio.run(_prime_hitl_test_graph(graph, config)).values[
            "exam_order_proposal"
        ]
        main.case_threads[proposal["case_id"]] = "api-hitl-thread"
        main.case_patients[proposal["case_id"]] = proposal["patient_id"]

        with (
            patch.object(main, "medical_graph", graph),
            patch.object(main, "CLINICIAN_APPROVAL_TOKEN", "test-token"),
        ):
            response = self.client.post(
                "/confirm_tool",
                json={
                    "patient_id": proposal["patient_id"],
                    "case_id": proposal["case_id"],
                    "proposal_id": proposal["proposal_id"],
                    "proposal_version": proposal["version"],
                    "payload_hash": proposal["payload_hash"],
                    "approved": True,
                },
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Clinician-Id": "doctor-001",
                    "X-Clinician-Role": "clinician",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("execute", response.text)
        self.assertEqual(
            graph.get_state(config).values["exam_order_execution_result"]["status"],
            "executed",
        )

    def test_api_rejects_tampered_proposal_binding(self):
        graph = _build_hitl_test_graph()
        config = {"configurable": {"thread_id": "api-tamper-thread"}}
        proposal = asyncio.run(_prime_hitl_test_graph(graph, config)).values[
            "exam_order_proposal"
        ]
        main.case_threads[proposal["case_id"]] = "api-tamper-thread"
        main.case_patients[proposal["case_id"]] = proposal["patient_id"]

        with (
            patch.object(main, "medical_graph", graph),
            patch.object(main, "CLINICIAN_APPROVAL_TOKEN", "test-token"),
        ):
            response = self.client.post(
                "/confirm_tool",
                json={
                    "patient_id": proposal["patient_id"],
                    "case_id": proposal["case_id"],
                    "proposal_id": proposal["proposal_id"],
                    "proposal_version": proposal["version"],
                    "payload_hash": "b" * 64,
                    "approved": True,
                },
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Clinician-Id": "doctor-001",
                    "X-Clinician-Role": "clinician",
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertNotIn("exam_order_execution_result", graph.get_state(config).values)


if __name__ == "__main__":
    unittest.main()
