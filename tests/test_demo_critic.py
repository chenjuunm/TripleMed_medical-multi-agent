"""Tests use fictional cases, identities, contacts, and credentials only.

Values that resemble personal or medical data are synthetic test fixtures and
do not represent real people or records.
"""

import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, patch

import graph


class DemoCriticTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "evidence": [{"evidence_id": "K001", "kind": "medical_knowledge",
                          "statement": "合成证据", "source": {"source_id": "DEMO", "is_demo": True}}],
            "synthesis_result": {"clinical_summary": "合成候选诊断", "abstain": False,
                "ranked_differential": [{"diagnosis": "合成候选", "supporting_evidence_ids": ["K001"]}],
                "recommended_actions": []},
        }
        self.raw = {"verdict": "revise", "severity": "medium", "issues": [{
            "type": "demo_source_limitation", "severity": "medium", "description": "来源为模拟数据"}],
            "safety_flags": [], "targeted_queries": ["真实指南"]}

    def review(self, enabled=True):
        with patch.object(graph, "CRITIC_DEMO_MODE", enabled), patch.object(
                graph, "invoke_json", AsyncMock(return_value=self.raw)) as call:
            result = asyncio.run(graph.critic_node(self.state))
        self.assertEqual(call.await_count, 1)
        return result

    def test_demo_source_warning_allows_full_report(self):
        result = self.review()
        self.assertEqual(result["critique_result"]["verdict"], "pass")
        self.assertEqual(result["critique_result"]["issues"], [])
        self.assertEqual(len(result["critique_result"]["warnings"]), 1)
        self.assertEqual(graph.route_after_critic(result), "finalize")
        report = asyncio.run(graph.finalize_node({**self.state, **result}))["final_report"]
        self.assertIn("合成候选", report)
        self.assertIn("模拟数据演示报告", report)
        self.assertIn("来源为模拟数据", report)

    def test_all_three_advisory_types_are_nonblocking(self):
        for kind in ("demo_source_limitation", "optional_information_gap", "presentation_only"):
            self.raw["issues"][0]["type"] = kind
            self.assertEqual(self.review()["critique_result"]["verdict"], "pass")

    def test_flag_off_restores_standard_review(self):
        self.assertEqual(self.review(False)["critique_result"]["verdict"], "revise")

    def test_real_unknown_and_mixed_sources_do_not_enable_demo(self):
        for flag in (False, None, "true"):
            self.state["evidence"][0]["source"]["is_demo"] = flag
            self.assertEqual(self.review()["critique_result"]["review_mode"], "standard")
        self.state["evidence"][0]["source"]["is_demo"] = True
        second = copy.deepcopy(self.state["evidence"][0])
        second["source"].pop("is_demo")
        self.state["evidence"].append(second)
        self.assertEqual(self.review()["critique_result"]["verdict"], "revise")

    def test_unknown_or_clinical_issues_are_not_downgraded(self):
        self.raw["issues"][0]["type"] = "allergy_conflict"
        self.assertEqual(self.review()["critique_result"]["verdict"], "revise")

    def test_invalid_evidence_id_still_blocks(self):
        self.state["synthesis_result"]["ranked_differential"][0]["supporting_evidence_ids"] = ["MISSING"]
        self.assertNotEqual(self.review()["critique_result"]["verdict"], "pass")

    def test_high_severity_and_safety_flags_still_block(self):
        self.raw["severity"] = "high"
        self.assertEqual(self.review()["critique_result"]["verdict"], "escalate")
        self.raw["severity"] = "medium"
        self.raw["safety_flags"] = ["risk"]
        self.assertEqual(self.review()["critique_result"]["verdict"], "escalate")

    def test_model_failure_is_not_a_pass(self):
        self.raw["verdict"] = "pass"
        self.raw["_fallback_reason"] = ""
        self.assertEqual(self.review()["critique_result"]["verdict"], "escalate")

    def test_abstention_and_no_evidence_are_not_overridden(self):
        self.state["synthesis_result"]["abstain"] = True
        self.assertEqual(self.review()["critique_result"]["verdict"], "escalate")
        self.state["synthesis_result"]["abstain"] = False
        self.state["evidence"] = []
        self.assertEqual(self.review()["critique_result"]["verdict"], "escalate")
