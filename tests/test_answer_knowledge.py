import asyncio
import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

import graph
import main
from answer_knowledge import AnswerKnowledgeBase, canonical_json, reusable_payload
import agents
from tools import _DEMO_ORDER_STORE


def draft(with_test=False):
    return {
        "case_id": "source-case", "patient_id": "SYNTHETIC-P001",
        "chief_complaint": "咽痛三天，无发热",
        "clinical_context": {"age": 30, "medications": [], "temperature": 36.5},
        "case_snapshot": {"chief_complaint": "咽痛三天，无发热"},
        "triage_result": {"urgency": "low", "route": "standard"},
        "work_plan": {},
        "evidence": [{"evidence_id": "K001", "kind": "medical_knowledge",
                      "statement": "合成来源，只用于测试", "source": {"source_id": "DEMO-001"}}],
        "synthesis_result": {"clinical_summary": "合成草案", "abstain": False,
            "ranked_differential": [{"diagnosis": "合成候选", "supporting_evidence_ids": ["K001"]}],
            "recommended_actions": ([{"type": "test", "action": "合成检查", "evidence_ids": ["K001"]}]
                                    if with_test else [])},
        "critique_result": {"verdict": "pass", "severity": "low", "model_fallback": False,
                            "issues": [], "safety_flags": []},
        "final_report": "合成完成报告", "messages": [], "audit_events": [],
    }


class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "answers.sqlite3"
        self.store = AnswerKnowledgeBase(self.path)
        self.state = draft(with_test=True)

    def save(self):
        s = self.state
        return self.store.vote(s["case_id"], s["patient_id"], s["chief_complaint"],
                               s["clinical_context"], "up", reusable_payload(s))

    def lookup(self, **changes):
        s = {**self.state, **changes}
        return self.store.lookup(s["patient_id"], s["chief_complaint"], s["clinical_context"])

    def test_cross_patient_loose_language_and_duration(self):
        self.save()
        for complaint in ("最近喉咙痛两天，无发烧", "没有发热，咽痛3天", "咽痛三天，无发热！"):
            with self.subTest(complaint=complaint):
                result = self.lookup(patient_id="B", chief_complaint=complaint, clinical_context={"age": 65})
                self.assertTrue(result)
                self.assertGreaterEqual(result[0]["score"], 0.82)

    def test_negation_new_symptoms_and_unrelated_complaints_do_not_directly_match(self):
        self.save()
        for complaint in ("咽痛三天，有发热", "咽痛三天，无发热，伴胸痛", "尿痛", "头痛"):
            self.assertEqual(self.lookup(chief_complaint=complaint), [])

    def test_full_clinical_content_is_preserved(self):
        self.state["final_report"] = "完整诊疗方案：合成候选；演示药物 500 mg；建议合成检查。\n复查计划保留。"
        self.save()
        plan = self.lookup()[0]["plan"]
        self.assertEqual(plan["report_text"], self.state["final_report"])
        self.assertEqual(plan["synthesis_result"], self.state["synthesis_result"])
        self.assertNotIn("critique_result", plan)
        self.assertNotIn("case_snapshot", plan)

    def test_identity_is_filtered_but_not_clinical_doses(self):
        self.state["clinical_context"].update({"name": "张三", "phone": "13800138000",
                                              "nested": {"address": "测试街道88号", "record_id": "MRN-SECRET"}})
        self.state["final_report"] = (
            "张三 SYNTHETIC-P001 source-case MRN-SECRET 测试街道88号\n"
            "电话：13800138000；邮箱：private@example.com；身份证：110101199001011234\n"
            "临床方案：合成候选，演示药物 500 mg。")
        self.save()
        with sqlite3.connect(self.path) as conn:
            dump = "\n".join(conn.iterdump())
        returned = canonical_json(self.lookup(patient_id="B"))
        for secret in ("张三", "13800138000", "private@example.com", "MRN-SECRET",
                       "测试街道88号", "110101199001011234", "SYNTHETIC-P001", "source-case"):
            self.assertNotIn(secret, dump)
            self.assertNotIn(secret, returned)
        self.assertIn("500 mg", returned)

    def test_private_ehr_locators_are_not_returned(self):
        self.state["evidence"][0].update({"query": {"patient_id": "SECRET"}})
        self.state["evidence"][0]["source"] = {"record_id": "RECORD-SECRET", "source_url": "https://ehr.test/patient/SECRET"}
        self.state["final_report"] = "历史证据 RECORD-SECRET https://ehr.test/patient/SECRET；合成检查保留。"
        self.save()
        result = canonical_json(self.lookup())
        self.assertNotIn("SECRET", result)
        self.assertNotIn("https://", result)
        self.assertIn("HISTORICAL-EVIDENCE", result)

    def test_ambiguous_unstructured_name_fails_closed_without_erasing_symptoms(self):
        self.state["final_report"] = "患者张三，建议合成检查。"
        with self.assertRaises(ValueError):
            self.save()
        self.state["clinical_context"]["name"] = "张三"
        self.save()
        self.assertNotIn("张三", canonical_json(self.lookup()))
        self.state["final_report"] = "患者咽痛，建议合成检查。"
        self.save()
        with sqlite3.connect(self.path) as conn:
            self.assertIn("患者咽痛", "\n".join(conn.iterdump()))

    def test_corruption_is_ignored(self):
        self.save()
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE liked_plans_v3 SET payload_json='{}'")
        self.assertEqual(self.lookup(), [])

    def test_rehashed_unknown_payload_fields_rejected(self):
        self.save()
        plan = self.lookup()[0]["plan"]
        plan["patient_name"] = "SECRET"
        encoded = canonical_json(plan)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE liked_plans_v3 SET id=?,payload_json=?,payload_hash=?", (digest, encoded, digest))
        self.assertEqual(self.lookup(), [])

    def test_reopen_duplicate_votes_and_downvote(self):
        self.save()
        self.save()
        s = self.state
        self.store.vote(s["case_id"], s["patient_id"], s["chief_complaint"], {}, "down")
        self.store = AnswerKnowledgeBase(self.path)
        self.assertTrue(self.lookup(patient_id="B"))
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM liked_plans_v3").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT vote FROM plan_feedback_v3").fetchone()[0], "down")

    def test_bad_admission_and_missing_traceability(self):
        for update in ({"model_fallback": True}, {"verdict": "revise"}, {"safety_flags": ["risk"]}, {"severity": "high"}):
            s = copy.deepcopy(self.state)
            s["critique_result"].update(update)
            with self.assertRaises(ValueError):
                self.store.vote("x", "P", s["chief_complaint"], {}, "up", s)
        self.state["synthesis_result"]["recommended_actions"][0]["evidence_ids"] = ["MISSING"]
        with self.assertRaises(ValueError):
            self.save()

    def test_legacy_full_liked_plan_is_privacy_filtered_on_read(self):
        self.state["final_report"] = "旧方案 SYNTHETIC-P001 source-case，合成检查保留。"
        encoded = canonical_json(reusable_payload(self.state))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE answers (id TEXT, patient_id TEXT, question TEXT, context_json TEXT, payload_json TEXT, payload_hash TEXT, updated_at TEXT, source_case_id TEXT)")
            conn.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?)", ("OLD", self.state["patient_id"],
                self.state["chief_complaint"], "{}", encoded, digest, "2026-09-07", "source-case"))
        result = self.lookup(patient_id="B")
        self.assertTrue(result)
        self.assertNotIn("SYNTHETIC-P001", canonical_json(result))
        self.assertNotIn("source-case", canonical_json(result))
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT payload_json FROM answers").fetchone()[0], encoded)

    def test_v2_topic_only_records_cannot_invent_old_plan(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE reference_patterns_v2 (payload_json TEXT)")
            conn.execute("INSERT INTO reference_patterns_v2 VALUES ('{}')")
        self.assertEqual(self.lookup(), [])

    def test_only_one_best_plan_returned_without_blending(self):
        self.save()
        self.state["final_report"] = "另一份完整方案"
        self.save()
        self.assertEqual(len(self.lookup()), 1)
        self.assertEqual(self.lookup(), self.lookup())

class KnowledgeApiAndGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = AnswerKnowledgeBase(Path(self.temp.name) / "answers.sqlite3")
        self.workflow = graph.workflow.compile(checkpointer=MemorySaver())
        self.addCleanup(patch.stopall)
        patch.object(main, "knowledge_base", self.store).start()
        patch.object(main, "medical_graph", self.workflow).start()
        patch.object(main, "CLINICIAN_APPROVAL_TOKEN", "synthetic-test-token").start()
        self.preflight = patch.object(main, "lm_studio_runtime_status", AsyncMock(return_value={
            "ready": True, "missing_models": []})).start()
        self.with_test = False
        self.critic_failure = False
        self.model_call = patch.object(graph, "invoke_json", AsyncMock(side_effect=self.synthetic_role)).start()
        self.search = patch.object(graph, "search_medical_guidelines").start()
        self.search.ainvoke = AsyncMock(return_value={"status": "found", "items": [{
            "content": "CURRENT EVIDENCE", "source": "合成指南", "source_id": "NEW-GUIDELINE", "is_demo": True}]})
        main.case_threads.clear()
        main.case_patients.clear()
        main.approval_claims.clear()
        _DEMO_ORDER_STORE.clear()
        self.client = TestClient(main.app)
        self.state = draft()

    async def synthetic_role(self, llm, system, payload, fallback, schema):
        if system == agents.INTAKE_TRIAGE_SYSTEM:
            raw = {"snapshot": {"chief_complaint": payload["chief_complaint"],
                    "demographics": payload["clinical_context"]},
                   "triage": {"urgency": "low", "route": "standard"}}
        elif system == agents.SUPERVISOR_SYSTEM:
            raw = {"case_complexity": "simple", "guideline_queries": ["本次指南查询"],
                   "specialist_perspectives": [{"role": "合成视角"}]}
        elif system == agents.SPECIALIST_SYSTEM:
            raw = {"certainty": "low"}
        elif system == agents.SYNTHESIS_SYSTEM:
            raw = copy.deepcopy(draft(self.with_test)["synthesis_result"])
            raw["clinical_summary"] = "CURRENT PATIENT NEW DRAFT"
        else:
            raw = {"verdict": "pass", "severity": "low"}
        result = schema.model_validate(raw).model_dump()
        if self.critic_failure and system == agents.CRITIC_SYSTEM:
            result["_fallback_reason"] = "synthetic timeout"
        return result

    def seed_source(self, with_test=False):
        self.state = draft(with_test)
        self.workflow.update_state({"configurable": {"thread_id": "source-thread"}}, self.state, as_node="finalize")
        main.case_threads[self.state["case_id"]] = "source-thread"
        main.case_patients[self.state["case_id"]] = self.state["patient_id"]

    def feedback(self, vote, case_id="source-case", **extra):
        return self.client.post("/answer_feedback", json={"case_id": case_id,
            "patient_id": self.state["patient_id"], "vote": vote, **extra})

    def chat(self, case_id="new-case", **extra):
        return self.client.post("/medical_chat", json={"case_id": case_id,
            "patient_id": self.state["patient_id"], "chief_complaint": self.state["chief_complaint"],
            "clinical_context": self.state["clinical_context"], **extra})

    def current_state(self, case_id="new-case"):
        return self.workflow.get_state({"configurable": {"thread_id": main.case_threads[case_id]}})

    def confirm(self, proposal, approved=True, **extra):
        body = {"case_id": proposal["case_id"], "patient_id": proposal["patient_id"],
                "proposal_id": proposal["proposal_id"], "proposal_version": proposal["version"],
                "payload_hash": proposal["payload_hash"], "approved": approved,
                "reason": "" if approved else "合成拒绝", **extra}
        return self.client.post("/confirm_tool", json=body, headers={
            "Authorization": "Bearer synthetic-test-token", "X-Clinician-Id": "synthetic-doctor",
            "X-Clinician-Role": "clinician"})

    def seed_and_like(self, with_test=False):
        self.seed_source(with_test)
        self.assertEqual(self.feedback("up").status_code, 200)

    def test_cross_patient_hit_skips_all_models_and_critic_but_requires_human(self):
        self.seed_and_like()
        self.preflight.return_value = {"ready": False, "missing_models": ["offline"]}
        self.model_call.side_effect = AssertionError("No model allowed on direct hit")
        response = self.chat(patient_id="CURRENT-PATIENT", chief_complaint="最近喉咙痛2天，无发烧",
                             clinical_context={"age": 65, "allergies": ["CURRENT-ALLERGY"]})
        self.assertEqual(response.status_code, 200)
        self.assertIn('"node": "reuse_liked_plan"', response.text)
        self.assertIn('"type": "interrupt"', response.text)
        self.assertNotIn("[DONE]", response.text)
        for node in ("intake_triage", "supervisor", "synthesis", "critic"):
            self.assertNotIn(f'"node": "{node}"', response.text)
        self.preflight.assert_not_awaited()
        self.model_call.assert_not_awaited()
        self.search.ainvoke.assert_not_awaited()
        values = self.current_state().values
        self.assertEqual(values["critique_result"]["verdict"], "not_run")
        self.assertIn("合成完成报告", values["final_report"])
        self.assertIn("CURRENT-PATIENT", values["final_report"])
        self.assertIn("CURRENT-ALLERGY", values["final_report"])
        self.assertNotIn("SYNTHETIC-P001", values["final_report"])
        self.assertNotIn("case_snapshot", values)
        proposal = values["exam_order_proposal"]
        self.assertEqual(proposal["patient_id"], "CURRENT-PATIENT")
        self.assertEqual(proposal["items"], [])
        self.assertFalse(_DEMO_ORDER_STORE)
        self.assertIn("[DONE]", self.confirm(proposal).text)
        self.assertEqual(self.current_state().values["exam_order_execution_result"]["status"], "review_approved")
        self.assertFalse(_DEMO_ORDER_STORE)

    def test_old_tests_are_preserved_but_only_execute_after_new_approval(self):
        self.seed_and_like(with_test=True)
        self.chat(patient_id="B")
        first = self.current_state().values["exam_order_proposal"]
        self.assertEqual(first["items"][0]["test"], "合成检查")
        self.assertFalse(_DEMO_ORDER_STORE)
        self.assertEqual(self.confirm(first, payload_hash="f"*64).status_code, 409)
        self.assertEqual(self.confirm(first).status_code, 200)
        self.assertEqual(len(_DEMO_ORDER_STORE), 1)
        self.assertEqual(self.confirm(first).status_code, 409)
        self.chat(case_id="next-case", patient_id="C")
        second = self.current_state("next-case").values["exam_order_proposal"]
        self.assertNotEqual(first["proposal_id"], second["proposal_id"])
        self.assertEqual(second["patient_id"], "C")
        self.assertEqual(self.current_state("next-case").next, ("await_clinician_approval",))

    def test_rejection_does_not_execute(self):
        self.seed_and_like(with_test=True)
        self.chat()
        self.confirm(self.current_state().values["exam_order_proposal"], approved=False)
        self.assertFalse(_DEMO_ORDER_STORE)
        self.assertEqual(self.current_state().values["exam_order_execution_result"]["status"], "rejected")

    def test_report_tampering_is_rejected(self):
        self.seed_and_like(with_test=True)
        self.chat()
        snapshot = self.current_state()
        proposal = snapshot.values["exam_order_proposal"]
        self.workflow.update_state(snapshot.config, {"final_report": "tampered"}, as_node="prepare_exam_order")
        self.confirm(proposal)
        self.assertEqual(self.current_state().values["approval_decision"]["decision"], "invalid")
        self.assertFalse(_DEMO_ORDER_STORE)

    def test_relike_records_same_plan_without_enrolling_current_unreviewed_case(self):
        self.seed_and_like()
        self.chat()
        response = self.feedback("up", case_id="new-case")
        self.assertEqual(response.status_code, 200)
        with sqlite3.connect(self.store.path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM liked_plans_v3").fetchone()[0], 1)
        self.assertEqual(self.current_state().values["critique_result"]["verdict"], "not_run")

    def test_downvote_does_not_disable_hit(self):
        self.seed_and_like()
        self.chat()
        self.assertEqual(self.feedback("down", case_id="new-case").status_code, 200)
        self.assertIn('"node": "reuse_liked_plan"', self.chat("next-case").text)

    def test_miss_runs_original_model_and_critic(self):
        self.seed_and_like()
        response = self.chat(chief_complaint="尿痛")
        self.assertIn('"node": "intake_triage"', response.text)
        self.assertIn('"node": "critic"', response.text)
        self.assertNotIn('"node": "reuse_liked_plan"', response.text)
        self.preflight.assert_awaited_once()

    def test_negation_conflict_offline_does_not_reuse(self):
        self.seed_and_like()
        self.preflight.return_value = {"ready": False, "missing_models": ["offline"]}
        self.assertEqual(self.chat(chief_complaint="咽痛三天，有发热").status_code, 503)
        self.model_call.assert_not_awaited()

    def test_deterministic_red_flags_skip_lookup(self):
        self.seed_and_like()
        with patch.object(self.store, "lookup", wraps=self.store.lookup) as lookup:
            response = self.chat(chief_complaint="大出血")
        lookup.assert_not_called()
        self.assertIn('"node": "emergency_response"', response.text)
        self.assertNotIn('"node": "reuse_liked_plan"', response.text)

    def test_database_read_failure_falls_back(self):
        with patch.object(self.store, "lookup", side_effect=sqlite3.OperationalError("busy")):
            response = self.chat()
        self.assertIn('"node": "critic"', response.text)
        self.preflight.assert_awaited_once()

    def test_feedback_authorization_and_write_error(self):
        self.seed_source()
        self.assertEqual(self.feedback("up", patient_id="B").status_code, 403)
        self.assertEqual(self.feedback("up", case_id="unknown").status_code, 404)
        self.assertEqual(self.feedback("wrong").status_code, 422)
        with patch.object(self.store, "vote", side_effect=sqlite3.OperationalError("busy")):
            self.assertEqual(self.feedback("up").status_code, 503)

    def test_graph_red_flag_guard_applies_even_if_plan_injected(self):
        from answer_knowledge import shareable_plan
        hit = {"plan": shareable_plan("A", self.state["chief_complaint"], {}, self.state)}
        route = graph.route_at_start({"chief_complaint": "大出血", "retrieved_plans": [hit]})
        self.assertEqual(route, "standard")

    def test_client_cannot_replace_stored_report(self):
        self.seed_source()
        self.assertEqual(self.feedback("up", answer="FORGED").status_code, 200)
        self.assertNotIn("FORGED", self.chat().text)

if __name__ == "__main__":
    unittest.main()
