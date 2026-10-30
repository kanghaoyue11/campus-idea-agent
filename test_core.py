import unittest
import json
from unittest.mock import patch

from core import LLMClient, MAX_RETRIES, retrieve_knowledge, run_orchestration, to_markdown
from scheduler import load_registry, select_agent


class FakeClient(LLMClient):
    def __init__(self, reviews):
        super().__init__("真实模型 API", "http://unused", "fake", "test-key")
        self.reviews = iter(reviews)
        self.calls = []

    def complete(self, system, user):
        self.calls.append(system)
        if "规划 Agent" in system:
            return "验证计划：访谈5位学生"
        if "方案 Agent" in system:
            return "# 方案\n## 用户痛点与证据\n待验证\n## 验证计划\n访谈\n## 成效指标\n5人\n## 风险与边界\n保护隐私"
        if "独立验证 Agent" in system:
            return next(self.reviews)
        if "方案修订 Agent" in system:
            return "# 修订方案\n## 用户痛点与证据\n待验证\n## 验证计划\n访谈\n## 成效指标\n5人\n## 风险与边界\n保护隐私"
        raise AssertionError("unexpected agent")


class CoreTests(unittest.TestCase):
    def test_local_knowledge_retrieval_is_bounded_and_relevant(self):
        hits = retrieve_knowledge("校园办事页面原型测试与任务完成时间", limit=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertLessEqual(len(hits), 2)
        self.assertIn("prototype_test", {x["id"] for x in hits})

    def test_scheduler_selects_capability_matched_agents(self):
        registry = load_registry()
        quick = select_agent("planner", "简单想法", False, registry)
        evidence = select_agent("planner", "访谈数据和验证指标", True, registry)
        self.assertEqual(quick.agent_id, "planner_fast")
        self.assertEqual(evidence.agent_id, "planner_evidence")
        reviewer = select_agent("reviewer", "结果指标", True, registry)
        self.assertEqual(reviewer.agent_id, "review_evidence")

    def test_mock_mode_labels_simulation_and_exports(self):
        result = run_orchestration("找不到办事信息", "新生", "", LLMClient("模拟模式（不调用真实模型）", "", "", ""))
        self.assertTrue(result["mode"].startswith("模拟"))
        self.assertTrue(result["review"]["passed"])
        self.assertIn("模拟模式", to_markdown(result))
        self.assertEqual(result["attempts"], 0)

    def test_real_mode_self_loop_retries_then_passes(self):
        client = FakeClient([
            '{"passed":false,"issues":["缺少风险章节"],"missing_sections":["风险与边界"],"summary":"需补充"}',
            '{"passed":true,"issues":[],"missing_sections":[],"summary":"复核通过"}',
        ])
        result = run_orchestration("找不到办事信息", "新生", "访谈3人", client)
        self.assertTrue(result["review"]["passed"])
        self.assertEqual(result["attempts"], 1)
        self.assertTrue(any(x["step"] == "Self-Loop 修订" for x in result["logs"]))

    def test_real_mode_stops_after_retry_limit(self):
        no_pass = '{"passed":false,"issues":["仍有问题"],"missing_sections":[],"summary":"未通过"}'
        client = FakeClient([no_pass] * (MAX_RETRIES + 1))
        result = run_orchestration("需求", "用户", "证据", client)
        self.assertFalse(result["review"]["passed"])
        self.assertEqual(result["attempts"], MAX_RETRIES)
        self.assertEqual(sum(1 for x in result["logs"] if x["step"] == "独立验证 Agent"), MAX_RETRIES + 1)


    def test_real_api_client_uses_openai_compatible_request(self):
        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "兼容接口响应"}}]}).encode()
        client = LLMClient("真实模型 API", "https://example.invalid/v1", "test-model", "test-key")
        with patch("core.urllib.request.urlopen", return_value=FakeResponse()) as open_url:
            answer = client.complete("system role", "user input")
        request = open_url.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(answer, "兼容接口响应")
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"][1]["content"], "user input")

    def test_missing_api_key_fails_clearly(self):
        client = LLMClient("真实模型 API", "https://api.openai.com/v1", "model", "")
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            client.complete("system", "user")


if __name__ == "__main__":
    unittest.main()
