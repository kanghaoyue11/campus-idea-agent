import threading
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import core
from core import (LLMClient, MAX_RETRIES, ROLE_MAX_ATTEMPTS, retrieve_knowledge,
                  run_orchestration, to_markdown)
from scheduler import load_registry, select_agent


class ScriptedClient(LLMClient):
    """按角色脚本化返回或抛异常，用于验证并发、重试与失败状态。"""

    def __init__(self, planner=None, solution=None, reviewer=None):
        super().__init__("真实模型 API", "http://unused", "fake", "test-key")
        # planner / solution 为可调用对象或异常；reviewer 为可迭代的 JSON 字符串序列。
        self.planner = planner
        self.solution = solution
        self.reviewer = iter(reviewer or [])
        self.calls = []

    def complete(self, system, user):
        self.calls.append(system)
        if "规划 Agent" in system:
            return self._resolve(self.planner)
        if "方案 Agent" in system:
            return self._resolve(self.solution)
        if "独立验证 Agent" in system:
            return next(self.reviewer)
        if "方案修订 Agent" in system:
            return "# 修订方案\n## 用户痛点与证据\n待验证\n## 验证计划\n访谈\n## 成效指标\n5人\n## 风险与边界\n保护隐私"
        raise AssertionError("unexpected agent")

    @staticmethod
    def _resolve(spec):
        if callable(spec):
            return spec()
        if isinstance(spec, Exception):
            raise spec
        return spec


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
    def setUp(self):
        # 将轨迹写入重定向到临时目录，避免测试污染项目真实的 runs/。
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_runs_dir = core.RUNS_DIR
        core.RUNS_DIR = Path(self._tmp.name)

    def tearDown(self):
        core.RUNS_DIR = self._orig_runs_dir
        self._tmp.cleanup()

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

    def test_single_role_retries_then_succeeds(self):
        """单角色首次失败、重试后成功：最终结果正确，且记录的是成功。"""
        calls = []

        def flaky_planner():
            calls.append("planner")
            if len(calls) == 1:
                raise RuntimeError("规划接口瞬时超时")
            return "验证计划：访谈5位学生"

        client = ScriptedClient(
            planner=flaky_planner,
            solution="# 方案\n## 用户痛点与证据\n待验证\n## 验证计划\n访谈\n## 成效指标\n5人\n## 风险与边界\n保护隐私",
            reviewer=['{"passed":true,"issues":[],"missing_sections":[],"summary":"通过"}'],
        )
        result = run_orchestration("需求", "用户", "证据", client)
        # 首次失败后重试成功，尝试了 2 次。
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["plan"], "验证计划：访谈5位学生")
        # 先记录首次异常，再记录重试成功。
        failures = [x for x in result["logs"] if x["step"] == "planner Agent" and x["status"] == "失败"]
        self.assertEqual(len(failures), 1)
        self.assertIn("第 1/3 次尝试异常", failures[0]["detail"])
        self.assertIn("规划接口瞬时超时", failures[0]["detail"])
        self.assertTrue(any(x["step"] == "planner Agent" and x["status"] == "完成"
                            for x in result["logs"]))

    def test_parallel_roles_run_on_shared_immutable_snapshot(self):
        """规划与方案基于同一份不可变输入快照并行运行，方案不读取规划输出。"""
        seen = []
        lock = threading.Lock()
        both_started = threading.Barrier(2)

        def planner():
            with lock:
                seen.append("planner")
            both_started.wait(timeout=2)
            return "验证计划：访谈5位学生"

        def solution():
            with lock:
                seen.append("solution")
            both_started.wait(timeout=2)
            return "# 方案\n## 用户痛点与证据\n待验证\n## 验证计划\n访谈\n## 成效指标\n5人\n## 风险与边界\n保护隐私"

        client = ScriptedClient(planner=planner, solution=solution,
                                reviewer=['{"passed":true,"issues":[],"missing_sections":[],"summary":"通过"}'])
        result = run_orchestration("找不到办事信息", "新生", "访谈3人", client)
        # 两个角色都被执行，且方案输出中不再包含规划的输出文本（独立快照）。
        self.assertIn("planner", seen)
        self.assertIn("solution", seen)
        self.assertNotIn("验证计划：访谈5位学生", result["solution"])
        self.assertIn("并行调度", {x["step"] for x in result["logs"]})

    def test_parallel_role_retries_up_to_limit_then_falls_back(self):
        """单角色连续失败：最多尝试 3 次，之后返回明确失败状态（安全空值）并记录异常。"""
        attempts = []

        def flaky_planner():
            attempts.append("planner")
            raise RuntimeError("规划接口超时")

        client = ScriptedClient(
            planner=flaky_planner,
            solution="# 方案\n## 用户痛点与证据\n待验证\n## 验证计划\n访谈\n## 成效指标\n5人\n## 风险与边界\n保护隐私",
            reviewer=['{"passed":true,"issues":[],"missing_sections":[],"summary":"通过"}'],
        )
        result = run_orchestration("需求", "用户", "证据", client)
        # 角色被调用 ROLE_MAX_ATTEMPTS 次（首次 + 2 次重试）。
        self.assertEqual(len(attempts), ROLE_MAX_ATTEMPTS)
        # 每次失败都有序号与原因；重试耗尽后返回安全空值，不伪装成通过。
        self.assertIn("生成失败", result["plan"])
        failures = [x for x in result["logs"] if x["step"] == "planner Agent" and x["status"] == "失败"]
        self.assertEqual(len(failures), ROLE_MAX_ATTEMPTS)
        self.assertEqual([x["detail"] for x in failures], [
            f"第 {attempt}/{ROLE_MAX_ATTEMPTS} 次尝试异常：RuntimeError: 规划接口超时"
            for attempt in range(1, ROLE_MAX_ATTEMPTS + 1)
        ])
        exhausted = [x for x in result["logs"] if x["step"] == "planner Agent" and x["status"] == "耗尽"]
        self.assertEqual(len(exhausted), 1)
        self.assertIn("规划接口超时", exhausted[0]["detail"])
        trace_path = next(Path(core.RUNS_DIR).glob("*.json"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        planner_trace = [x for x in trace["roles"] if x["role"] == "planner Agent"]
        self.assertEqual([x["status"] for x in planner_trace], ["failed"] * ROLE_MAX_ATTEMPTS + ["fallback"])
        self.assertEqual([x["retries"] for x in planner_trace], [0, 1, 2, 2])
        self.assertTrue(all("规划接口超时" in x["error"] for x in planner_trace))

    def test_successful_role_result_not_overwritten_by_failure(self):
        """一个角色成功、另一个角色失败时，成功结果不被覆盖，失败返回明确失败状态（安全空值）。"""
        client = ScriptedClient(
            planner=lambda: "验证计划：访谈5位学生",
            solution=RuntimeError("方案接口 500"),
            reviewer=['{"passed":true,"issues":[],"missing_sections":[],"summary":"通过"}'],
        )
        result = run_orchestration("需求", "用户", "证据", client)
        self.assertEqual(result["plan"], "验证计划：访谈5位学生")
        self.assertIn("生成失败", result["solution"])
        # 各次失败原因与最终耗尽状态都写入日志，不伪装成通过。
        failures = [x for x in result["logs"] if x["step"] == "solution Agent" and x["status"] == "失败"]
        self.assertEqual(len(failures), ROLE_MAX_ATTEMPTS)
        self.assertTrue(all("方案接口 500" in x["detail"] for x in failures))
        self.assertTrue(any(x["step"] == "solution Agent" and x["status"] == "耗尽"
                            and "500" in x["detail"] for x in result["logs"]))


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

    def test_trace_file_generated_correctly(self):
        """每次编排都生成结构化轨迹，包含 task_id 与每个角色的时间线信息。"""
        result = run_orchestration("找不到办事信息", "新生", "",
                                   LLMClient("模拟模式（不调用真实模型）", "", "", ""))
        files = list(Path(self._tmp.name).glob("*.json"))
        self.assertEqual(len(files), 1)
        trace = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertIn("task_id", trace)
        self.assertTrue(trace["task_id"])
        self.assertIn("roles", trace)
        self.assertGreaterEqual(len(trace["roles"]), 1)
        # 每个角色条目具备时间线所需字段。
        for role in trace["roles"]:
            self.assertIn("role", role)
            self.assertIn("status", role)
            self.assertIn("duration_ms", role)
        self.assertTrue(result["mode"].startswith("模拟"))

    def test_trace_write_failure_does_not_break_main_flow(self):
        """轨迹写入抛异常时，编排主流程仍成功返回正常结果。"""
        with patch.object(core, "_persist_trace", side_effect=OSError("磁盘只读")):
            result = run_orchestration("找不到办事信息", "新生", "",
                                       LLMClient("模拟模式（不调用真实模型）", "", "", ""))
        self.assertTrue(result["mode"].startswith("模拟"))
        self.assertTrue(result["review"]["passed"])
        self.assertIn("plan", result)
        self.assertIn("solution", result)


if __name__ == "__main__":
    unittest.main()
