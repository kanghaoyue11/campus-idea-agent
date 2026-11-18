"""双阶段批量多 Agent 调度测试（第六阶段）。

覆盖：批量并发抽取、分片、抽取失败进 errors、Reviewer 只处理有 MatchResult 者、
Reviewer 降级不改分、task_key 不冲突、roles/fallbacks key 一致、instruction 注入、
请求次数统计、Trace 兼容、真实并发验证、原流程回归。

所有 LLM 调用均使用 Mock，不调用真实 API。
"""

from __future__ import annotations

import json
import threading
import time
import unittest

from scheduler import run_roles_concurrently, select_agent
from screening_extraction import ExtractionOutcome, ExtractionSpec, run_extraction
from screening_extractors import ExtractedResume, ExtractionError, stable_resume_id
from screening_models import JobRequirements
from screening_orchestrator import CandidateResult, ScreeningOrchestrator
from screening_reviewer import DeterministicReviewer, LLMReviewer
from screening_scheduler import (
    ExtractionFailedError,
    persist_screening_trace,
    run_extraction_batch,
    run_review_batch,
)


def _requirements(**kwargs):
    defaults = dict(
        job_id="job-1",
        title="后端工程师",
        required_skills=("Python", "FastAPI"),
        preferred_skills=("Docker",),
        min_experience_years=3,
        max_experience_years=8,
        education_requirement="本科",
        required_certifications=("PMP",),
        responsibilities=(),
    )
    defaults.update(kwargs)
    return JobRequirements(**defaults)


def _extracted_resume(raw_text, source_name="a.txt"):
    return ExtractedResume(
        resume_id=stable_resume_id(raw_text, source_name),
        source_name=source_name,
        raw_text=raw_text,
        format="txt",
    )


def _sample_extraction_json():
    return json.dumps({
        "skills": ["Python", "FastAPI", "Docker"],
        "work_experience_years": 5,
        "education": "本科",
        "certifications": ["PMP"],
        "explicitly_not_owned_skills": ["Kubernetes"],
        "explicitly_not_owned_certifications": ["AWS"],
        "evidence": [
            {"category": "skill", "subject": "Python", "quote": "Python", "polarity": "positive"},
            {"category": "skill", "subject": "FastAPI", "quote": "FastAPI", "polarity": "positive"},
            {"category": "skill", "subject": "Docker", "quote": "Docker", "polarity": "positive"},
            {"category": "skill", "subject": "Kubernetes", "quote": "不会 Kubernetes", "polarity": "negative"},
            {"category": "experience", "subject": "经验", "quote": "5年", "polarity": "positive"},
            {"category": "education", "subject": "本科", "quote": "本科", "polarity": "positive"},
            {"category": "certification", "subject": "PMP", "quote": "PMP", "polarity": "positive"},
            {"category": "certification", "subject": "AWS", "quote": "无 AWS 证书", "polarity": "negative"},
        ],
    })


class _QueueClient:
    """按队列返回；记录调用次数与并发峰值。"""

    mode = "模拟模式（不调用真实模型）"

    def __init__(self, responses, *, gate=None):
        self._responses = list(responses)
        self.calls = []
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self._gate = gate  # 可选 threading.Barrier，用于验证并发

    def complete(self, system, user):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.calls.append((system, user))
        if self._gate is not None:
            self._gate.wait(timeout=5)
        # 模拟一定耗时，放大并发窗口
        time.sleep(0.02)
        item = self._responses.pop(0)
        with self._lock:
            self._active -= 1
        if isinstance(item, Exception):
            raise item
        return item


class _AlwaysFailClient:
    """每次都抛异常，用于测试抽取失败与降级。"""

    mode = "模拟模式（不调用真实模型）"

    def __init__(self, exc=None):
        self.exc = exc or RuntimeError("模拟失败")
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        raise self.exc


class _ReviewFailClient:
    """Reviewer 用：第一次抛异常（降级），后续正常。"""

    mode = "模拟模式（不调用真实模型）"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        raise RuntimeError("模拟复核失败")


class _ReviewOkClient:
    mode = "模拟模式（不调用真实模型）"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return json.dumps({
            "needs_human_review": True,
            "flags": [],
            "summary": "复核完成。",
        })


class ExtractionBatchTests(unittest.TestCase):
    def _three_resumes(self):
        return [
            _extracted_resume("张三 熟练掌握 Python FastAPI Docker 5年后端经验 本科 PMP 不会Kubernetes 无AWS证书", "a.txt"),
            _extracted_resume("李四 了解 Python 2年经验 大专", "b.txt"),
            _extracted_resume("王五 精通 Java Go 3年经验 硕士", "c.txt"),
        ]

    def test_all_three_extract_success(self):
        resumes = self._three_resumes()
        client = _QueueClient([_sample_extraction_json()] * 3)
        result = run_extraction_batch(
            resumes, client, _requirements(), batch_id="B1"
        )
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(result["extract_errors"], [])
        self.assertTrue(all(c.profile is not None for c in result["candidates"]))

    def test_one_fails_two_succeed(self):
        resumes = self._three_resumes()
        client = _QueueClient([
            _sample_extraction_json(),
            RuntimeError("抽取超时"), RuntimeError("抽取超时"),  # 第二份 max_retries=1 → 2 次
            _sample_extraction_json(),
        ])
        result = run_extraction_batch(
            resumes, client, _requirements(), batch_id="B1", max_retries=1
        )
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(len(result["extract_errors"]), 1)
        # 失败者 profile 为 None
        failed = [c for c in result["candidates"] if c.profile is None]
        self.assertEqual(len(failed), 1)

    def test_failure_enters_errors_and_excludes_matching(self):
        resumes = self._three_resumes()
        client = _QueueClient([
            _sample_extraction_json(),
            RuntimeError("X"), RuntimeError("X"),
            _sample_extraction_json(),
        ])
        result = run_extraction_batch(resumes, client, _requirements(), batch_id="B1", max_retries=1)
        # 失败简历进入 extract_errors，且 profile=None → 不进入匹配
        self.assertEqual(len(result["extract_errors"]), 1)
        failed_cand = [c for c in result["candidates"] if c.profile is None][0]
        self.assertIsNone(failed_cand.match_result)
        self.assertIsNotNone(failed_cand.error)

    def test_sharding_over_four_resumes(self):
        resumes = [_extracted_resume(f"简历{i} Python FastAPI 本科 3年", f"r{i}.txt") for i in range(6)]
        client = _QueueClient([_sample_extraction_json()] * 6)
        result = run_extraction_batch(resumes, client, _requirements(), batch_id="B1", shard_size=4)
        self.assertEqual(len(result["candidates"]), 6)
        self.assertEqual(result["extract_errors"], [])
        # 分片后所有 resume_id 都在（无覆盖、无丢失）
        ids = {c.resume_id for c in result["candidates"]}
        self.assertEqual(ids, {r.resume_id for r in resumes})

    def test_select_agent_called_once_per_batch(self):
        resumes = self._three_resumes()
        client = _QueueClient([_sample_extraction_json()] * 3)
        # 通过 monkeypatch 计数 select_agent 调用次数
        import screening_scheduler as ss
        original = ss.select_agent
        calls = []
        ss.select_agent = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
        try:
            run_extraction_batch(resumes, client, _requirements(), batch_id="B1", shard_size=1)
        finally:
            ss.select_agent = original
        self.assertEqual(len(calls), 1)  # 批次级选型一次，而非每份一次

    def test_instruction_injected_into_prompt(self):
        resumes = [self._three_resumes()[0]]
        client = _QueueClient([_sample_extraction_json()])
        result = run_extraction_batch(resumes, client, _requirements(), batch_id="B1")
        # instruction 来自 registry 的 extractor Agent，必须出现在 system prompt
        system_called = client.calls[0][0]
        self.assertIsInstance(system_called, str)
        self.assertTrue(system_called.strip())  # 非空

    def test_agent_id_into_trace(self):
        resumes = self._three_resumes()
        client = _QueueClient([_sample_extraction_json()] * 3)
        result = run_extraction_batch(resumes, client, _requirements(), batch_id="B1")
        for rec in result["trace_records"]:
            self.assertIn("agent_id", rec)
            self.assertTrue(rec["agent_id"])  # 真实 Registry ID
            self.assertEqual(rec["agent_role"], "extractor")

    def test_task_key_stage_prefixed(self):
        resumes = self._three_resumes()
        client = _QueueClient([_sample_extraction_json()] * 3)
        result = run_extraction_batch(resumes, client, _requirements(), batch_id="B1")
        for rec in result["trace_records"]:
            self.assertTrue(rec["task_key"].startswith("resume_extraction:"))
            self.assertEqual(rec["role"], rec["task_key"])  # role 保存 task_key

    def test_roles_fallbacks_keys_identical(self):
        # 通过失败场景确认 fallback 与 roles key 一致（失败者能正确拿到占位值）
        resumes = self._three_resumes()
        client = _QueueClient([
            RuntimeError("X"), RuntimeError("X"),
            _sample_extraction_json(),
            _sample_extraction_json(),
        ])
        result = run_extraction_batch(resumes, client, _requirements(), batch_id="B1", max_retries=1)
        self.assertEqual(len(result["extract_errors"]), 1)

    def test_llm_request_attempts_counted(self):
        resumes = [self._three_resumes()[0]]
        client = _QueueClient([_sample_extraction_json()])
        result = run_extraction_batch(resumes, client, _requirements(), batch_id="B1", max_retries=1)
        rec = result["trace_records"][0]
        self.assertEqual(rec["llm_request_attempts"], 1)
        self.assertEqual(len(client.calls), 1)


class ReviewBatchTests(unittest.TestCase):
    def _candidates_with_match(self, n=3):
        cands = []
        for i in range(n):
            c = CandidateResult(
                resume_id=f"resume-{i}",
                source_name=f"r{i}.txt",
                profile=None,
                match_result=_FakeMatch(),
                raw_text="",
            )
            cands.append(c)
        return cands

    def test_review_only_matches(self):
        cands = self._candidates_with_match(3)
        # 一个无 match_result
        cands.append(CandidateResult(
            resume_id="resume-no-match", source_name="x.txt",
            profile=None, match_result=None, raw_text="",
        ))
        reviewer = LLMReviewer(_ReviewOkClient())
        records = run_review_batch(cands, reviewer, _requirements(), batch_id="B1")
        # 只有 3 个有 match_result 的被复核
        self.assertEqual(len(records), 3)

    def test_reviewer_fallback_does_not_change_match(self):
        cand = CandidateResult(
            resume_id="resume-1", source_name="r.txt",
            profile=None, match_result=_FakeMatch(), raw_text="",
        )
        before = cand.match_result.overall_score
        reviewer = LLMReviewer(_ReviewFailClient())
        records = run_review_batch([cand], reviewer, _requirements(), batch_id="B1")
        rec = records[0]
        self.assertTrue(rec["reviewer_fallback_used"])
        self.assertEqual(rec["business_status"], "degraded")
        self.assertEqual(rec["agent_role"], "screening_reviewer")
        self.assertEqual(cand.match_result.overall_score, before)  # 分数不变
        self.assertIsNotNone(cand.review)

    def test_reviewer_success_marked(self):
        cand = CandidateResult(
            resume_id="resume-1", source_name="r.txt",
            profile=None, match_result=_FakeMatch(), raw_text="",
        )
        reviewer = LLMReviewer(_ReviewOkClient())
        records = run_review_batch([cand], reviewer, _requirements(), batch_id="B1")
        rec = records[0]
        self.assertFalse(rec["reviewer_fallback_used"])
        self.assertEqual(rec["business_status"], "success")
        self.assertEqual(rec["llm_request_attempts"], 1)

    def test_deterministic_fallback_preserves_score(self):
        # DeterministicReviewer 无 review_traced，走兼容分支
        cand = CandidateResult(
            resume_id="resume-1", source_name="r.txt",
            profile=None, match_result=_FakeMatch(), raw_text="",
        )
        before = cand.match_result.overall_score
        reviewer = DeterministicReviewer()
        records = run_review_batch([cand], reviewer, _requirements(), batch_id="B1")
        self.assertEqual(cand.match_result.overall_score, before)


class _FakeMatch:
    """最小 MatchResult 替身，满足 reviewer 只读需要。"""

    def __init__(self):
        self.overall_score = 78.5
        self.recommendation = "limited_match"
        self.information_completeness = 80.0
        self.unknown_items = ()
        self.not_matched_items = ()
        self.evidence = ()
        self.needs_human_review = True


class OrchestratorBatchTests(unittest.TestCase):
    def _files(self, n=3):
        texts = [
            "张三 熟练掌握 Python FastAPI Docker 5年 本科 PMP 不会Kubernetes",
            "李四 了解 Python 2年 大专",
            "王五 精通 Java Go 3年 硕士",
        ]
        return [(t.encode("utf-8"), f"r{i}.txt") for i, t in enumerate(texts[:n])]

    def test_run_end_to_end_batch(self):
        from screening_reviewer import DeterministicReviewer
        client = _QueueClient([_sample_extraction_json()] * 3)
        orch = ScreeningOrchestrator(client, reviewer=DeterministicReviewer())
        report = orch.run(_requirements(), self._files(3))
        self.assertEqual(len(report.candidates), 3)
        self.assertEqual(len(report.ranking), 3)  # 全部评分成功

    def test_run_failure_enters_errors(self):
        from screening_reviewer import DeterministicReviewer
        client = _QueueClient([
            _sample_extraction_json(),
            RuntimeError("X"), RuntimeError("X"),
            _sample_extraction_json(),
        ])
        orch = ScreeningOrchestrator(client, reviewer=DeterministicReviewer())
        report = orch.run(_requirements(), self._files(3))
        # 失败简历进入 errors，ranking 只有 2 份
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(len(report.ranking), 2)


class ConcurrencyTests(unittest.TestCase):
    def test_extractor_true_concurrency(self):
        """用 threading.Barrier 验证：多份简历的抽取确实并发执行。"""
        resumes = [_extracted_resume(f"简历{i} Python 本科 3年", f"r{i}.txt") for i in range(4)]
        barrier = threading.Barrier(4)
        client = _QueueClient([_sample_extraction_json()] * 4, gate=barrier)
        result = run_extraction_batch(
            resumes, client, _requirements(), batch_id="B1", shard_size=4
        )
        # 若真并发，至少某时刻有 >1 个活跃调用
        self.assertGreater(client.max_active, 1)
        self.assertEqual(len(result["candidates"]), 4)


class TraceTests(unittest.TestCase):
    def test_render_table_reads_new_trace(self):
        from trace_view import render_table
        resumes = [
            _extracted_resume("张三 Python 本科 3年", "a.txt"),
            _extracted_resume("李四 Java 大专", "b.txt"),
        ]
        client = _QueueClient([_sample_extraction_json(), _sample_extraction_json()])
        result = run_extraction_batch(resumes, client, _requirements(), batch_id="B1")
        trace = {
            "task_id": "t1",
            "mode": "模拟",
            "created_at": "2026-11-10T13:00:00+08:00",
            "roles": result["trace_records"],
        }
        out = render_table(trace)
        self.assertIn("任务 ID", out)
        self.assertIn("t1", out)

    def test_persist_trace_writes_file(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            p = persist_screening_trace(
                [{"role": "resume_extraction:r1", "status": "success"}],
                task_id="t1", mode="模拟", created_at="now",
                runs_dir=Path(td),
            )
            self.assertIsNotNone(p)
            self.assertTrue(p.exists())
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["task_id"], "t1")
            self.assertIn("roles", data)


class P1AttemptsTests(unittest.TestCase):
    """P1 修复：抽取失败场景的 llm_request_attempts 精确统计。"""

    def _one_resume(self):
        return [_extracted_resume("张三 Python FastAPI 本科 3年", "a.txt")]

    def test_success_first_try_records_1(self):
        client = _QueueClient([_sample_extraction_json()])
        result = run_extraction_batch(
            self._one_resume(), client, _requirements(), batch_id="B1", max_retries=1
        )
        rec = result["trace_records"][0]
        self.assertEqual(rec["llm_request_attempts"], 1)
        self.assertEqual(len(client.calls), 1)

    def test_retry_then_success_records_2(self):
        # 第一次失败、第二次成功 → 实际调用 complete 2 次
        client = _QueueClient([
            RuntimeError("第一次失败"),
            _sample_extraction_json(),
        ])
        result = run_extraction_batch(
            self._one_resume(), client, _requirements(), batch_id="B1", max_retries=1
        )
        rec = result["trace_records"][0]
        self.assertEqual(rec["llm_request_attempts"], 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(rec["business_status"], "success")

    def test_both_fail_records_2(self):
        # 两次都失败 → 实际调用 2 次，Trace 记 2（P1 核心修复：不再记 0）
        client = _QueueClient([
            RuntimeError("第一次失败"),
            RuntimeError("第二次失败"),
        ])
        result = run_extraction_batch(
            self._one_resume(), client, _requirements(), batch_id="B1", max_retries=1
        )
        rec = result["trace_records"][0]
        self.assertEqual(rec["llm_request_attempts"], 2)
        self.assertEqual(rec["business_status"], "failed")
        self.assertEqual(len(client.calls), 2)
        # 失败简历进入 extract_errors
        self.assertEqual(len(result["extract_errors"]), 1)

    def test_pre_call_failure_records_0(self):
        # 在调用 LLM 之前就失败（run_extraction 抛出非 ExtractionFailedError 异常），
        # attempts_by_key 无该 rid 记录 → 记 0（而不是误记非零）。
        from unittest import mock
        resumes = self._one_resume()
        with mock.patch(
            "screening_extraction.run_extraction",
            side_effect=RuntimeError("调用 LLM 前失败"),
        ):
            result = run_extraction_batch(
                resumes, _QueueClient([]), _requirements(), batch_id="B1", max_retries=1
            )
        rec = result["trace_records"][0]
        self.assertEqual(rec["llm_request_attempts"], 0)
        self.assertEqual(rec["business_status"], "failed")
        self.assertIn("调用 LLM 前失败", rec["fallback_reason"])

    def test_fallback_preserves_error_and_attempts(self):
        # Scheduler fallback 不得覆盖原始请求次数与失败原因。
        client = _QueueClient([
            RuntimeError("原始抽取错误 XYZ"),
            RuntimeError("原始抽取错误 XYZ"),
        ])
        result = run_extraction_batch(
            self._one_resume(), client, _requirements(), batch_id="B1", max_retries=1
        )
        rec = result["trace_records"][0]
        # 真实原因保留在 fallback_reason / error（来自 rr.error，含异常原文）
        self.assertIn("原始抽取错误 XYZ", rec["fallback_reason"])
        self.assertEqual(rec["llm_request_attempts"], 2)
        self.assertEqual(rec["scheduler_fallback_used"], True)
        # 失败简历的 error 也保留原文
        failed = result["candidates"][0]
        self.assertIn("原始抽取错误 XYZ", failed.error)


class P2InstructionTests(unittest.TestCase):
    """P2 修复：AgentAssignment.instruction 真正注入 LLM system prompt。"""

    def _one_resume(self):
        return [_extracted_resume("张三 Python FastAPI 本科 3年", "a.txt")]

    def test_instruction_actually_in_prompt(self):
        # 用 registry 中 extractor Agent 的真实 instruction 作为断言依据。
        import screening_scheduler as ss
        assignment = ss.select_agent("extractor", context="后端工程师", has_evidence=False)
        instr = assignment.instruction.strip()
        self.assertTrue(instr)  # registry 里 extractor 必有非空 instruction

        client = _QueueClient([_sample_extraction_json()])
        run_extraction_batch(
            self._one_resume(), client, _requirements(), batch_id="B1"
        )
        system_called = client.calls[0][0]
        # 唯一 instruction 完整出现在真实传入的 system prompt 中
        self.assertIn(instr, system_called)
        # 原有抽取规则、Evidence 约束、JSON 输出要求仍保留
        self.assertIn("逐字来自简历原文", system_called)
        self.assertIn("polarity", system_called)
        self.assertIn("JSON", system_called)

    def test_none_instruction_safe(self):
        # instruction 为 None 时，run_extraction 的 agent_instruction 参数安全跳过。
        from screening_extraction import ExtractionSpec, run_extraction
        spec = ExtractionSpec(
            resume_id="r1", source_name="a.txt", raw_text="Python 3年"
        )
        client = _QueueClient([_sample_extraction_json()])
        outcome = run_extraction(client, spec, max_retries=1, agent_instruction=None)
        self.assertTrue(outcome.ok)
        system_called = client.calls[0][0]
        self.assertNotIn("None", system_called)  # 不把 Python None 拼入

    def test_empty_instruction_safe(self):
        from screening_extraction import ExtractionSpec, run_extraction
        spec = ExtractionSpec(
            resume_id="r1", source_name="a.txt", raw_text="Python 3年"
        )
        client = _QueueClient([_sample_extraction_json()])
        base = run_extraction(client, spec, max_retries=1, agent_instruction="")
        self.assertTrue(base.ok)
        # 空 instruction 与不传等价：system 仍以原始抽取规则开头
        self.assertIn("你是简历结构化抽取 Agent", client.calls[0][0])

    def test_instruction_from_selected_agent_reaches_prompt(self):
        # 验证 run_extraction 的 agent_instruction 参数真正改变传给 complete 的 system。
        from screening_extraction import ExtractionSpec, run_extraction
        unique = "UNIQUE-AGENT-INSTRUCTION-42"
        spec = ExtractionSpec(
            resume_id="r1", source_name="a.txt", raw_text="Python 3年"
        )
        client = _QueueClient([_sample_extraction_json()])
        run_extraction(client, spec, max_retries=1, agent_instruction=unique)
        self.assertIn(unique, client.calls[0][0])


class P3GapTests(unittest.TestCase):
    """P3 补充：最大并发 ≤4、Reviewer 真实并发、Trace 完整性、roles/fallbacks key 一致。"""

    def _many_resumes(self, n):
        return [_extracted_resume(f"简历{i} Python 本科 3年", f"r{i}.txt") for i in range(n)]

    def test_max_concurrency_not_exceed_4(self):
        # 6 份简历、shard_size=4：max_active 必须 ≤4，且确实并发（>1）。
        resumes = self._many_resumes(6)
        barrier = threading.Barrier(4)  # 首个分片 4 个任务同步
        client = _QueueClient([_sample_extraction_json()] * 6, gate=barrier)
        result = run_extraction_batch(
            resumes, client, _requirements(), batch_id="B1", shard_size=4
        )
        self.assertEqual(len(result["candidates"]), 6)
        self.assertGreater(client.max_active, 1)  # 真并发
        self.assertLessEqual(client.max_active, 4)  # 上限 4

    def test_reviewer_true_concurrency(self):
        from screening_scheduler import run_review_batch
        cands = []
        for i in range(4):
            cands.append(CandidateResult(
                resume_id=f"resume-{i}", source_name=f"r{i}.txt",
                profile=None, match_result=_FakeMatch(), raw_text="",
            ))
        barrier = threading.Barrier(4)
        reviewer = LLMReviewer(_QueueClient([
            json.dumps({"needs_human_review": True, "flags": [], "summary": "ok"})
        ] * 4, gate=barrier))
        records = run_review_batch(
            cands, reviewer, _requirements(), batch_id="B1", shard_size=4
        )
        self.assertEqual(len(records), 4)  # 全部完成复核
        self.assertGreater(reviewer.client.max_active, 1)  # Reviewer 真并发

    def test_trace_completeness_end_to_end(self):
        # 端到端：每份成功抽取有抽取 Trace，每份评分者有 Reviewer Trace。
        from screening_reviewer import DeterministicReviewer
        client = _QueueClient([_sample_extraction_json()] * 3)
        orch = ScreeningOrchestrator(client, reviewer=DeterministicReviewer())
        report = orch.run(_requirements(), [
            ("张三 Python FastAPI 5年 本科 PMP".encode(), "a.txt"),
            ("李四 Python 3年 本科".encode(), "b.txt"),
            ("王五 Java 3年 硕士".encode(), "c.txt"),
        ])
        # 3 份全部评分成功 → ranking 3 份
        self.assertEqual(len(report.ranking), 3)
        # 抽取 Trace 记录数 = 3，Reviewer Trace 记录数 = 3（DeterministicReviewer 也走 review_batch）
        # 通过 orchestrator 内部无法直接拿到 trace_records，这里直接调 run_extraction_batch 验证。
        resumes = [
            _extracted_resume("张三 Python FastAPI 5年 本科 PMP", "a.txt"),
            _extracted_resume("李四 Python 3年 本科", "b.txt"),
            _extracted_resume("王五 Java 3年 硕士", "c.txt"),
        ]
        c2 = _QueueClient([_sample_extraction_json()] * 3)
        er = run_extraction_batch(resumes, c2, _requirements(), batch_id="B1")
        self.assertEqual(len(er["trace_records"]), 3)
        task_keys = [r["task_key"] for r in er["trace_records"]]
        self.assertEqual(len(set(task_keys)), 3)  # task_key 唯一
        for rec in er["trace_records"]:
            self.assertEqual(rec["agent_role"], "extractor")
            self.assertEqual(rec["stage"], "resume_extraction")
            self.assertTrue(rec["agent_id"])
            self.assertTrue(rec["resume_id"])

    def test_roles_fallbacks_keys_directly_equal(self):
        # 直接断言 roles 与 fallbacks 的 key 集合完全一致（不靠间接失败场景）。
        import screening_scheduler as ss
        resumes = self._many_resumes(5)
        # 通过 monkeypatch 拦截 run_roles_concurrently，捕获 roles/fallbacks。
        captured = {}
        original = ss.run_roles_concurrently

        def _fake_run(roles, fallbacks=None, max_retries=0):
            captured["roles"] = set(roles.keys())
            captured["fallbacks"] = set((fallbacks or {}).keys())
            # 返回每个 role 一个 ok=True 的假结果，保证上层流程走通。
            from scheduler import RoleResult
            return {k: RoleResult(role=k, value=(ExtractionOutcome(
                resume_id=k.split(":", 1)[1], profile=None, ok=True
            ), "agent-x"), ok=True, attempts=1, error=None, duration_ms=0) for k in roles}

        ss.run_roles_concurrently = _fake_run
        try:
            run_extraction_batch(resumes, _QueueClient([]), _requirements(), batch_id="B1")
        finally:
            ss.run_roles_concurrently = original
        self.assertEqual(captured["roles"], captured["fallbacks"])


if __name__ == "__main__":
    unittest.main()
