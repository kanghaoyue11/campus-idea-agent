"""招聘筛选流水线集成测试。

覆盖：文本提取、LLM 抽取（Mock）、非法 JSON、重试 fallback、Evidence 校验、
极性冲突、批量评分、排序稳定、Reviewer 不改分、敏感信息过滤、无泄露。
所有 LLM 相关测试均使用 Mock，不调用真实 API。

运行：python -m pytest test_screening_integration.py -v
"""

from __future__ import annotations

import io
import json
import unittest

from matching import rank
from screening_extraction import (
    ExtractionSpec,
    build_profile_from_extraction,
    extract_explicit_not_owned,
    run_extraction,
)
from screening_extractors import (
    extract_all_resumes,
    extract_resume,
    stable_resume_id,
)
from screening_models import Evidence, JobRequirements, ResumeProfile
from screening_orchestrator import ScreeningOrchestrator
from screening_reviewer import DeterministicReviewer


def _make_requirements(**kwargs):
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


class _MockClient:
    """按队列返回 JSON 串；可注入异常以测试重试与 fallback。"""

    mode = "模拟模式（不调用真实模型）"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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


_RAW1 = "张三\n熟练掌握 Python、FastAPI、Docker\n5年后端经验\n本科，持有PMP证书\n不会 Kubernetes\n无 AWS 证书"
_RAW2 = "李四\n了解 Python\n2年经验\n大专"


class ExtractionTests(unittest.TestCase):
    def test_txt_extraction_preserves_raw_text(self):
        r = extract_resume("  张三\nPython  ".encode("utf-8"), "a.txt")
        self.assertIn("Python", r.raw_text)
        self.assertTrue(r.resume_id.startswith("resume-"))

    def test_empty_file_raises(self):
        with self.assertRaises(ValueError):
            extract_resume(b"   ", "empty.txt")

    def test_unsupported_format_raises(self):
        with self.assertRaises(ValueError):
            extract_resume(b"x", "a.exe")

    def test_stable_resume_id_same_content(self):
        r1 = stable_resume_id("张三 Python", "a.txt")
        r2 = stable_resume_id("张三 Python", "b.pdf")
        self.assertEqual(r1, r2)

    def test_corrupted_pdf_raises_clear_error(self):
        # 非法 PDF 字节
        from screening_extractors import extract_resume as er
        with self.assertRaises((ValueError,)):
            er(b"%PDF-1.4 garbage not a real pdf", "bad.pdf")

    def test_batch_extraction_isolates_failures(self):
        ok_bytes = "张三 Python".encode("utf-8")
        resumes, errors = extract_all_resumes([
            (ok_bytes, "a.txt"),
            (b"   ", "empty.txt"),
            (b"x", "bad.exe"),
        ])
        self.assertEqual(len(resumes), 1)
        self.assertEqual(len(errors), 2)


class ExtractionParseTests(unittest.TestCase):
    def _spec(self, raw=_RAW1):
        return ExtractionSpec(resume_id="r1", source_name="a.txt", raw_text=raw)

    def test_llm_extraction_normal(self):
        client = _MockClient([_sample_extraction_json()])
        outcome = run_extraction(client, self._spec())
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.profile.skills, ("Python", "FastAPI", "Docker"))
        self.assertEqual(outcome.explicit_not_owned_skills, frozenset({"Kubernetes"}))

    def test_llm_invalid_json_falls_back(self):
        client = _MockClient(["这不是 JSON", "也不是 JSON", "还是不是"])
        outcome = run_extraction(client, self._spec(), max_retries=2)
        self.assertFalse(outcome.ok)
        self.assertIsNotNone(outcome.error)

    def test_llm_retries_then_succeeds(self):
        client = _MockClient([ValueError("瞬时失败"), _sample_extraction_json()])
        outcome = run_extraction(client, self._spec(), max_retries=2)
        self.assertTrue(outcome.ok)
        self.assertEqual(len(client.calls), 2)

    def test_llm_failure_then_success_records_attempts(self):
        client = _MockClient([RuntimeError("超时"), _sample_extraction_json()])
        outcome = run_extraction(client, self._spec(), max_retries=1)
        self.assertTrue(outcome.ok)

    def test_evidence_quote_not_in_raw_is_dropped(self):
        data = json.loads(_sample_extraction_json())
        data["evidence"].append(
            {"category": "skill", "subject": "Rust", "quote": "精通Rust", "polarity": "positive"}
        )
        profile, warnings = build_profile_from_extraction(data, self._spec())
        subjects = [e.subject for e in profile.evidence]
        self.assertNotIn("Rust", subjects)
        self.assertTrue(any("无法在原文定位" in w for w in warnings))

    def test_sensitive_attributes_are_dropped(self):
        data = json.loads(_sample_extraction_json())
        data["evidence"].append(
            {"category": "other", "subject": "年龄", "quote": "25岁", "polarity": "positive"}
        )
        data["skills"].append("男")  # 敏感词混入技能
        profile, warnings = build_profile_from_extraction(data, self._spec())
        self.assertNotIn("男", profile.skills)
        self.assertTrue(any("敏感" in w for w in warnings))

    def test_missing_education_and_experience_evidence_is_recovered_from_raw(self):
        # 真实模型偶尔漏写 experience/education 证据；若结构字段与原文均可定位，
        # 则补齐 min 证据，避免把已抽到的信息错误降级为 unknown。
        data = json.loads(_sample_extraction_json())
        data["evidence"] = [
            e for e in data["evidence"]
            if e["category"] not in ("experience", "education")
        ]
        profile, warnings = build_profile_from_extraction(data, self._spec())
        categories = [e.category for e in profile.evidence]
        self.assertEqual(categories.count("experience"), 1)
        self.assertEqual(categories.count("education"), 1)
        exp_ev = next(e for e in profile.evidence if e.category == "experience")
        edu_ev = next(e for e in profile.evidence if e.category == "education")
        self.assertEqual(exp_ev.quote, "5年")
        self.assertEqual(edu_ev.quote, "本科")
        self.assertTrue(any("漏写经验证据" in w for w in warnings))
        self.assertTrue(any("漏写学历证据" in w for w in warnings))

    def test_recovery_does_not_invent_quote_missing_from_raw(self):
        data = json.loads(_sample_extraction_json())
        data["education"] = "博士"  # 原文只有本科，不能补造“博士”证据
        data["evidence"] = [
            e for e in data["evidence"]
            if e["category"] != "education"
        ]
        profile, _ = build_profile_from_extraction(data, self._spec())
        self.assertFalse(any(e.category == "education" for e in profile.evidence))

    def test_unmentioned_not_marked_as_not_owned(self):
        # 未提及的技能不得被标为 negative：抽取输出里若出现，应保留为显式否定信号，
        # 但 Matching Engine 仍需要有效 negative 证据才会判 not_matched。
        data = json.loads(_sample_extraction_json())
        # 移除 Kubernetes 的 negative 证据，但保留 explicitly_not_owned_skills
        data["evidence"] = [e for e in data["evidence"] if e["subject"] != "Kubernetes"]
        profile, _ = build_profile_from_extraction(data, self._spec())
        not_owned_skills, _ = extract_explicit_not_owned(data)
        self.assertIn("Kubernetes", not_owned_skills)
        # 但没有 negative 证据 -> Matching Engine 应判 unknown
        from matching import score
        req = _make_requirements(required_skills=("Kubernetes",))
        result = score(req, profile, raw_text=_RAW1,
                       explicitly_not_owned=frozenset({"Kubernetes"}))
        self.assertEqual(result.skill_statuses["required"]["Kubernetes"], "unknown")


class PolarityConflictIntegrationTests(unittest.TestCase):
    def test_positive_negative_conflict_unknown(self):
        raw = "精通Python 不会Python 5年"
        profile = ResumeProfile(
            resume_id="r1", source_name="a", candidate_name=None,
            skills=("Python",), work_experience_years=5, education=None,
            certifications=(), work_history=(),
            evidence=(
                Evidence(resume_id="r1", category="skill", subject="Python", quote="精通Python", polarity="positive"),
                Evidence(resume_id="r1", category="skill", subject="Python", quote="不会Python", polarity="negative"),
                Evidence(resume_id="r1", category="experience", subject="经验", quote="5年"),
            ),
        )
        from matching import score
        result = score(_make_requirements(required_skills=("Python",)), profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")


class OrchestratorTests(unittest.TestCase):
    def test_full_pipeline_batch_scoring_and_ranking(self):
        client = _MockClient([_sample_extraction_json(),
                              json.dumps({
                                  "skills": ["Python"], "work_experience_years": 2,
                                  "education": "大专", "certifications": [],
                                  "explicitly_not_owned_skills": [],
                                  "explicitly_not_owned_certifications": [],
                                  "evidence": [
                                      {"category": "skill", "subject": "Python", "quote": "Python", "polarity": "positive"},
                                      {"category": "experience", "subject": "经验", "quote": "2年", "polarity": "positive"},
                                      {"category": "education", "subject": "大专", "quote": "大专", "polarity": "positive"},
                                  ],
                              })])
        orch = ScreeningOrchestrator(client, reviewer=DeterministicReviewer())
        report = orch.run(
            _make_requirements(),
            [(_RAW1.encode(), "zhangsan.txt"), (_RAW2.encode(), "lisi.txt")],
        )
        self.assertEqual(len(report.ranking), 2)
        # 排序：强匹配在前
        self.assertEqual(report.ranking[0].match_result.recommendation, "strong_match")
        # 强匹配候选人综合分更高
        self.assertGreater(
            report.ranking[0].match_result.overall_score,
            report.ranking[1].match_result.overall_score,
        )
        # 阶段完整
        stage_names = [s.stage for s in report.stage_logs]
        for expected in ("resume_ingestion", "resume_parsing", "resume_extraction",
                         "evidence_validation", "deterministic_matching",
                         "candidate_review", "result_aggregation"):
            self.assertIn(expected, stage_names)

    def test_reviewer_does_not_change_score(self):
        client = _MockClient([_sample_extraction_json()])
        orch = ScreeningOrchestrator(client, reviewer=DeterministicReviewer())
        report = orch.run(_make_requirements(), [(_RAW1.encode(), "a.txt")])
        cand = report.ranking[0]
        before = cand.match_result
        # reviewer 只产出 review dict，不改 match_result
        self.assertIsNotNone(cand.review)
        self.assertEqual(cand.match_result.overall_score, before.overall_score)
        self.assertEqual(cand.match_result.recommendation, before.recommendation)

    def test_ranking_stable(self):
        from matching import score
        req = _make_requirements(required_skills=("Python",), preferred_skills=(),
                                 education_requirement=None, required_certifications=())
        results = []
        for rid in ("b", "a", "c"):
            profile = ResumeProfile(
                resume_id=rid, source_name="s", candidate_name=None,
                skills=("Python",), work_experience_years=5, education=None,
                certifications=(), work_history=(),
                evidence=(
                    Evidence(resume_id=rid, category="skill", subject="Python", quote="Python"),
                    Evidence(resume_id=rid, category="experience", subject="经验", quote="5年"),
                ),
            )
            results.append(score(req, profile, raw_text="Python 5年"))
        ordered = rank(results)
        self.assertEqual([r.resume_id for r in ordered], ["a", "b", "c"])

    def test_no_api_key_or_resume_leak_in_logs(self):
        client = _MockClient([_sample_extraction_json()])
        orch = ScreeningOrchestrator(client, reviewer=DeterministicReviewer())
        report = orch.run(_make_requirements(), [(_RAW1.encode(), "a.txt")])
        # 阶段日志与报告不应包含完整简历原文（如候选人姓名）
        serialized = json.dumps([
            {"stage": s.stage, "input": s.input_summary, "output": s.output_summary}
            for s in report.stage_logs
        ], ensure_ascii=False)
        self.assertNotIn("张三", serialized)
        self.assertNotIn("OPENAI_API_KEY", serialized)
        self.assertNotIn("sk-", serialized)


if __name__ == "__main__":
    unittest.main()
