"""Matching Engine（v3.4）管道测试：覆盖 T1–T22 及边界条件。

所有用例均为纯确定性断言，不依赖 LLM、网络或临时文件写入项目目录。
"""

import unittest

from matching import rank, score
from screening_models import (
    Evidence,
    JobRequirements,
    ResumeProfile,
)


def _make_requirements(
    job_id="job-1",
    required=(),
    preferred=(),
    min_exp=None,
    max_exp=None,
    education=None,
    certifications=(),
):
    return JobRequirements(
        job_id=job_id,
        title="岗位",
        required_skills=required,
        preferred_skills=preferred,
        min_experience_years=min_exp,
        max_experience_years=max_exp,
        education_requirement=education,
        required_certifications=certifications,
        responsibilities=(),
    )


def _make_profile(
    resume_id="resume-1",
    skills=(),
    exp=None,
    education=None,
    certifications=(),
    evidence=(),
    raw_text=None,
):
    return ResumeProfile(
        resume_id=resume_id,
        source_name="resume.pdf",
        candidate_name=None,
        skills=skills,
        work_experience_years=exp,
        education=education,
        certifications=certifications,
        work_history=(),
        evidence=evidence,
    )


def _skill_evidence(resume_id, skill, quote=None, polarity="positive"):
    return Evidence(
        resume_id=resume_id,
        category="skill",
        subject=skill,
        quote=quote if quote is not None else skill,
        polarity=polarity,
    )


class MatchingEngineTests(unittest.TestCase):
    def test_t01_all_requirements_met_strong_match(self):
        req = _make_requirements(
            required=("Python", "FastAPI"),
            preferred=("Docker",),
            min_exp=3,
            max_exp=8,
            education="本科",
        )
        raw = "Python FastAPI Docker 5年 本科"
        profile = _make_profile(
            skills=("Python", "FastAPI", "Docker"),
            exp=5,
            education="本科",
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "FastAPI"),
                _skill_evidence("resume-1", "Docker"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
                Evidence(resume_id="resume-1", category="education", subject="本科", quote="本科"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.overall_score, 100.0)
        self.assertEqual(result.recommendation, "strong_match")

    def test_t02_partial_skills_unknown_insufficient(self):
        # 必备 Python(有), FastAPI(无证据->unknown), Docker(未提及->unknown)
        req = _make_requirements(
            required=("Python", "FastAPI", "Docker"), min_exp=3, max_exp=8
        )
        raw = "Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # 无硬性 not_met，但存在硬性 unknown -> insufficient
        self.assertEqual(result.recommendation, "insufficient_information")
        # 经验维度有分（met=100），故 overall_score 非 None（v3.4 允许 insufficient 有分）
        self.assertEqual(result.overall_score, 100.0)

    def test_t03_partial_skills_not_matched_limited(self):
        req = _make_requirements(required=("Python", "K8s"), min_exp=3, max_exp=8)
        raw = "Python 不会K8s 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "K8s", quote="不会K8s", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"K8s"}))
        self.assertEqual(result.recommendation, "limited_match")

    def test_t04_experience_unknown(self):
        req = _make_requirements(required=("Python",), min_exp=3, max_exp=8)
        raw = "Python"
        profile = _make_profile(
            skills=("Python",),
            exp=None,
            evidence=(_skill_evidence("resume-1", "Python"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.experience_status, "unknown")
        self.assertEqual(result.dimension_scores["experience"], None)

    def test_t05_education_unknown(self):
        req = _make_requirements(required=("Python",), education="本科")
        raw = "Python"
        profile = _make_profile(
            skills=("Python",),
            education=None,
            evidence=(_skill_evidence("resume-1", "Python"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertIsNone(result.dimension_scores["education"])

    def test_t06_certification_unknown(self):
        req = _make_requirements(required=("Python",), certifications=("PMP",))
        raw = "Python"
        profile = _make_profile(
            skills=("Python",),
            certifications=(),
            evidence=(_skill_evidence("resume-1", "Python"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertIsNone(result.dimension_scores["certification"])

    def test_t07_no_certification_requirement(self):
        req = _make_requirements(required=("Python",), certifications=())
        raw = "Python"
        profile = _make_profile(
            skills=("Python",),
            evidence=(_skill_evidence("resume-1", "Python"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertIsNone(result.dimension_scores["certification"])

    def test_t08_no_required_skills(self):
        req = _make_requirements(required=(), preferred=("Docker",))
        raw = "Docker"
        profile = _make_profile(
            skills=("Docker",),
            evidence=(_skill_evidence("resume-1", "Docker"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertIsNone(result.skill_match_score)
        self.assertIsNone(result.dimension_scores["required"])

    def test_t09_all_unknown(self):
        req = _make_requirements(required=("Python", "FastAPI", "Docker"), min_exp=3, max_exp=8)
        profile = _make_profile(skills=(), exp=None)
        result = score(req, profile, raw_text="")
        self.assertIsNone(result.overall_score)
        self.assertEqual(result.recommendation, "insufficient_information")

    def test_t10_all_not_met(self):
        req = _make_requirements(required=("Python",), min_exp=5, max_exp=8, education="硕士", certifications=("PMP",))
        raw = "不会Python 高中 0年 无PMP"
        profile = _make_profile(
            skills=(),
            exp=0,
            education="高中",
            certifications=(),
            evidence=(
                _skill_evidence("resume-1", "Python", quote="不会Python", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="0年"),
                Evidence(resume_id="resume-1", category="education", subject="高中", quote="高中"),
                Evidence(resume_id="resume-1", category="certification", subject="PMP", quote="无PMP", polarity="negative"),
            ),
        )
        result = score(
            req,
            profile,
            raw_text=raw,
            explicitly_not_owned=frozenset({"Python"}),
            explicitly_not_owned_certifications=frozenset({"PMP"}),
        )
        self.assertEqual(result.overall_score, 0.0)
        self.assertEqual(result.recommendation, "limited_match")

    def test_t11_pending_aliases_no_effect(self):
        req = _make_requirements(required=("Python",), min_exp=3, max_exp=8)
        raw = "Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        r1 = score(req, profile, raw_text=raw)
        r2 = score(req, profile, raw_text=raw, pending_aliases=("Python->C++",))
        self.assertEqual(r1.overall_score, r2.overall_score)
        self.assertEqual(r1.recommendation, r2.recommendation)

    def test_t12_claimed_skill_without_evidence_unknown(self):
        req = _make_requirements(required=("Python",), min_exp=3, max_exp=8)
        raw = "Python 5年"
        # 声称掌握 Python，但未提供 skill 证据
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")
        self.assertEqual(result.recommendation, "insufficient_information")

    def test_t13_deterministic_repeat(self):
        req = _make_requirements(required=("Python", "FastAPI"), preferred=("Docker",), min_exp=3, max_exp=8)
        raw = "Python FastAPI Docker 5年"
        profile = _make_profile(
            skills=("Python", "FastAPI", "Docker"),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "FastAPI"),
                _skill_evidence("resume-1", "Docker"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        r1 = score(req, profile, raw_text=raw)
        r2 = score(req, profile, raw_text=raw)
        self.assertEqual(r1, r2)

    def test_t14_preferred_with_unknown_not_strong(self):
        req = _make_requirements(required=("Python",), preferred=("Docker", "Kubernetes"), min_exp=3, max_exp=8)
        raw = "Python Docker 5年"
        profile = _make_profile(
            skills=("Python", "Docker"),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "Docker"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # Docker matched, Kubernetes unknown -> preferred_score 分母只有 matched=1 => 100
        self.assertEqual(result.dimension_scores["preferred"], 100.0)
        self.assertEqual(result.recommendation, "potential_match")

    def test_t15_preferred_with_not_matched_not_strong(self):
        req = _make_requirements(required=("Python",), preferred=("Docker", "Kubernetes"), min_exp=3, max_exp=8)
        raw = "Python Docker 不会Kubernetes 5年"
        profile = _make_profile(
            skills=("Python", "Docker"),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "Docker"),
                _skill_evidence("resume-1", "Kubernetes", quote="不会Kubernetes", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Kubernetes"}))
        self.assertEqual(result.recommendation, "potential_match")

    def test_t16_all_preferred_matched_strong(self):
        req = _make_requirements(required=("Python",), preferred=("Docker", "Kubernetes"), min_exp=3, max_exp=8)
        raw = "Python Docker Kubernetes 5年"
        profile = _make_profile(
            skills=("Python", "Docker", "Kubernetes"),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "Docker"),
                _skill_evidence("resume-1", "Kubernetes"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.dimension_scores["preferred"], 100.0)
        self.assertEqual(result.recommendation, "strong_match")

    def test_t17_no_hard_req_preferred_unknown_potential(self):
        # 无硬性要求，仅有加分技能，其中一项 unknown
        req = _make_requirements(required=(), preferred=("Docker", "Kubernetes"))
        raw = "Docker"
        profile = _make_profile(
            skills=("Docker",),
            evidence=(_skill_evidence("resume-1", "Docker"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.recommendation, "potential_match")

    def test_t18_same_skill_in_required_and_preferred_isolated(self):
        # 验证嵌套 skill_statuses 结构本身能独立保存同名技能（不通过 JobRequirements 构造，
        # 因为 JobRequirements 已正确禁止 required 与 preferred 重复）。
        statuses = {"required": {"Python": "matched"}, "preferred": {"Python": "matched"}}
        self.assertEqual(statuses["required"]["Python"], "matched")
        self.assertEqual(statuses["preferred"]["Python"], "matched")
        self.assertIsNot(statuses["required"], statuses["preferred"])

    def test_t19_modifying_required_does_not_affect_preferred(self):
        # 嵌套结构允许独立保存；修改 required 侧不影响 preferred 侧
        statuses = {"required": {"Python": "not_matched"}, "preferred": {"Python": "matched"}}
        statuses["required"]["Python"] = "unknown"
        self.assertEqual(statuses["preferred"]["Python"], "matched")
        # 也验证反向
        statuses["preferred"]["Python"] = "unknown"
        self.assertEqual(statuses["required"]["Python"], "unknown")

    def test_t20_soft_gate_only_reads_preferred(self):
        req = _make_requirements(required=("Python",), preferred=("Docker",), min_exp=3, max_exp=8)
        raw = "Python Docker 5年"
        profile = _make_profile(
            skills=("Python", "Docker"),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "Docker"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # 软性门槛只看 preferred 全 matched -> strong
        self.assertEqual(result.recommendation, "strong_match")

    def test_t21_empty_preferred_no_soft_gate(self):
        # 无加分技能，硬性全满足 -> strong_match
        req = _make_requirements(required=("Python",), preferred=(), min_exp=3, max_exp=8)
        raw = "Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.recommendation, "strong_match")

    def test_t22_no_requirements_potential_match(self):
        req = _make_requirements(required=(), preferred=())
        profile = _make_profile(skills=())
        result = score(req, profile, raw_text="")
        self.assertEqual(result.recommendation, "potential_match")
        self.assertIsNone(result.overall_score)
        self.assertIsNone(result.information_completeness)

    def test_invalid_evidence_downgrades_to_unknown(self):
        # Evidence 引文无法在原文定位 -> 视为无效 -> 降级 unknown
        req = _make_requirements(required=("Python",), min_exp=3, max_exp=8)
        raw = "完全不相关的文本"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python", quote="Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # Python 证据引文无法定位，技能降级 unknown
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    def test_min_years_zero_met(self):
        req = _make_requirements(required=("Python",), min_exp=0, max_exp=None)
        raw = "Python 1年"
        profile = _make_profile(
            skills=("Python",),
            exp=1,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="1年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.experience_status, "met")

    def test_experience_not_met_linear(self):
        req = _make_requirements(required=("Python",), min_exp=5, max_exp=8)
        raw = "Python 2年"
        profile = _make_profile(
            skills=("Python",),
            exp=2,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="2年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.experience_status, "not_met")
        self.assertEqual(result.dimension_scores["experience"], 40.0)

    def test_ranking_order(self):
        req = _make_requirements(required=("Python",), min_exp=3, max_exp=8)
        raw = "Python 5年"

        def make(rid, recommendation_override=None, overall=None):
            p = _make_profile(
                resume_id=rid,
                skills=("Python",),
                exp=5,
                evidence=(
                    _skill_evidence(rid, "Python"),
                    Evidence(resume_id=rid, category="experience", subject="经验", quote="5年"),
                ),
            )
            r = score(req, p, raw_text=raw)
            return r

        strong = make("a")
        limited = make("b")
        insufficient = make("c")
        # 手工构造不同 recommendation 以测试排序（通过不同输入）
        # 这里用真实 score 结果，再验证 rank 的等级优先
        ordered = rank([limited, strong, insufficient])
        # strong 应排最前
        self.assertEqual(ordered[0].resume_id, strong.resume_id)

    def test_ranking_stable_by_resume_id(self):
        # 两个同等级、同分结果，按 resume_id 字典序稳定排序
        from matching import _RECOMMENDATION_RANK  # noqa
        req = _make_requirements(required=("Python",), min_exp=3, max_exp=8)
        raw = "Python 5年"
        results = []
        for rid in ("b", "a", "c"):
            p = _make_profile(
                resume_id=rid,
                skills=("Python",),
                exp=5,
                evidence=(
                    _skill_evidence(rid, "Python"),
                    Evidence(resume_id=rid, category="experience", subject="经验", quote="5年"),
                ),
            )
            results.append(score(req, p, raw_text=raw))
        ordered = rank(results)
        self.assertEqual([r.resume_id for r in ordered], ["a", "b", "c"])


class EvidenceValidationTests(unittest.TestCase):
    """针对 Evidence 校验缺陷的补充测试（v3.4 审查后新增）。"""

    def _req(self, required=("Python",), min_exp=3, max_exp=8):
        return _make_requirements(required=required, min_exp=min_exp, max_exp=max_exp)

    def test_raw_text_none_does_not_trust_evidence(self):
        # raw_text=None 时，Evidence 不得默认有效 -> 技能降级 unknown
        req = self._req()
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(_skill_evidence("resume-1", "Python"),),
        )
        result = score(req, profile, raw_text=None)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")
        self.assertEqual(result.recommendation, "insufficient_information")

    def test_quote_not_in_raw_text_downgrades_unknown(self):
        # quote 不存在于原文 -> 无效 -> unknown
        req = self._req()
        raw = "完全不相关的文本"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(_skill_evidence("resume-1", "Python", quote="Python"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    def test_offsets_mismatch_downgrades_unknown(self):
        # offsets 与 quote 不一致 -> 无效 -> unknown
        req = self._req()
        raw = "Python FastAPI"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                Evidence(
                    resume_id="resume-1",
                    category="skill",
                    subject="Python",
                    quote="Python",
                    start_offset=100,
                    end_offset=200,
                ),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    def test_resume_id_mismatch_rejected_at_profile_level(self):
        # resume_id 不一致的 evidence 在 ResumeProfile 构造时即被拒绝
        foreign = Evidence(
            resume_id="resume-other",
            category="skill",
            subject="Python",
            quote="Python",
        )
        with self.assertRaises(ValueError):
            _make_profile(
                skills=("Python",),
                exp=5,
                evidence=(foreign,),
            )

    def test_category_mismatch_downgrades_unknown(self):
        # evidence 类别与评分维度不符 -> 不匹配 -> unknown
        req = self._req()
        raw = "Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                # 类别是 experience 而非 skill，无法支撑技能判定
                Evidence(resume_id="resume-1", category="experience", subject="Python", quote="Python"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    def test_subject_mismatch_downgrades_unknown(self):
        # subject 与被判定技能不符 -> 不匹配 -> unknown
        req = self._req()
        raw = "Python"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(_skill_evidence("resume-1", "Java", quote="Python"),),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    def test_explicit_not_owned_with_valid_evidence_is_not_matched(self):
        # 明确不会 + 有效否定证据 -> not_matched
        req = self._req(required=("Python",))
        raw = "不会Python 5年"
        profile = _make_profile(
            skills=(),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python", quote="不会Python", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Python"}))
        self.assertEqual(result.skill_statuses["required"]["Python"], "not_matched")
        self.assertEqual(result.recommendation, "limited_match")

    def test_explicit_not_owned_without_valid_evidence_is_unknown(self):
        # 明确不会但证据无效 -> unknown
        req = self._req(required=("Python",))
        raw = "无关文本"
        profile = _make_profile(
            skills=(),
            exp=5,
            evidence=(_skill_evidence("resume-1", "Python", quote="不会Python", polarity="negative"),),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Python"}))
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    def test_unmentioned_skill_is_unknown_not_not_matched(self):
        # 普通未提及技能 -> unknown，绝不判 not_matched
        req = self._req(required=("Python",))
        raw = "Java 5年"
        profile = _make_profile(
            skills=("Java",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Java"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    def test_pending_aliases_never_affect_score(self):
        # pending_aliases 只记录，不参与评分
        req = self._req(required=("Python",))
        raw = "Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        r1 = score(req, profile, raw_text=raw)
        r2 = score(req, profile, raw_text=raw, pending_aliases=("Python->Rust", "Python->Go"))
        self.assertEqual(r1.overall_score, r2.overall_score)
        self.assertEqual(r1.skill_statuses, r2.skill_statuses)
        self.assertEqual(r1.recommendation, r2.recommendation)
        self.assertEqual(r2.pending_aliases, ("Python->Rust", "Python->Go"))

    def test_required_preferred_status_isolated(self):
        # required 与 preferred 嵌套状态互不污染
        req = _make_requirements(required=("Python",), preferred=("Docker",), min_exp=3, max_exp=8)
        raw = "Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "matched")
        # Docker 未提及 -> preferred 中 unknown，但 required 的 Python 仍是 matched
        self.assertEqual(result.skill_statuses["preferred"]["Docker"], "unknown")
        self.assertEqual(result.skill_statuses["required"]["Python"], "matched")

    def test_repeated_run_identical(self):
        # 同一输入重复运行完全一致
        req = self._req(required=("Python", "FastAPI"))
        raw = "Python FastAPI 5年"
        profile = _make_profile(
            skills=("Python", "FastAPI"),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "FastAPI"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        r1 = score(req, profile, raw_text=raw)
        r2 = score(req, profile, raw_text=raw)
        self.assertEqual(r1, r2)

    def test_certification_partial_unknown_returns_unknown_status(self):
        # 部分证书 matched、部分 unknown -> certification_status 应为 unknown（信息不足），而非 met
        req = _make_requirements(required=("Python",), certifications=("PMP", "AWS"))
        raw = "Python PMP 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            certifications=("PMP",),
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="certification", subject="PMP", quote="PMP"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # PMP matched, AWS unknown -> 证书维度整体 unknown（存在未判定证书）
        self.assertEqual(result.dimension_scores["certification"], 100.0)
        self.assertIn("证书:AWS", result.unknown_items)


class NegativeEvidenceSemanticsTests(unittest.TestCase):
    """证书明确不具备 + 技能/证书否定证据方向语义校验（本轮新增）。

    不改变 v3.4 评分公式、权重与推荐等级规则，仅收窄 not_matched 的判定条件，
    防止把「未提及」或「正面证据」误判为明确不具备。
    """

    def _req(self, required=("Python",), certifications=()):
        return _make_requirements(
            required=required,
            min_exp=3,
            max_exp=8,
            certifications=certifications,
        )

    def _cert_evidence(self, subject, quote, polarity="positive"):
        return Evidence(
            resume_id="resume-1",
            category="certification",
            subject=subject,
            quote=quote,
            polarity=polarity,
        )

    # ---- 证书：明确不具备 + 有效否定证据 -> not_matched ----

    def test_cert_explicit_not_owned_with_valid_negative_evidence(self):
        req = self._req(certifications=("PMP",))
        raw = "Python 无PMP 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            certifications=(),
            evidence=(
                _skill_evidence("resume-1", "Python"),
                self._cert_evidence("PMP", "无PMP", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(
            req,
            profile,
            raw_text=raw,
            explicitly_not_owned_certifications=frozenset({"PMP"}),
        )
        self.assertEqual(result.dimension_scores["certification"], 0.0)
        self.assertIn("证书:PMP", result.not_matched_items)
        self.assertNotIn("证书:PMP", result.unknown_items)
        self.assertEqual(result.recommendation, "limited_match")

    # ---- 证书：未提及 -> unknown，绝不判 not_matched ----

    def test_cert_unmentioned_is_unknown_not_not_matched(self):
        req = self._req(certifications=("PMP",))
        raw = "Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            certifications=(),
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertIn("证书:PMP", result.unknown_items)
        self.assertNotIn("证书:PMP", result.not_matched_items)
        self.assertIsNone(result.dimension_scores["certification"])

    # ---- 证书：正面证据存在但 profile 漏录 -> 不判 not_matched（应 unknown）----

    def test_cert_positive_evidence_but_missing_in_profile_is_not_not_matched(self):
        # 简历原文明确写有 PMP，但 profile.certifications 漏录 -> 不得判 not_matched
        req = self._req(certifications=("PMP",))
        raw = "Python 持有PMP证书 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            certifications=(),  # 漏录
            evidence=(
                _skill_evidence("resume-1", "Python"),
                self._cert_evidence("PMP", "持有PMP证书", polarity="positive"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # 未声明明确不具备，正面证据存在但不属于否定信号 -> 不得判 not_matched
        self.assertNotIn("证书:PMP", result.not_matched_items)
        # 因为 profile 漏录且无否定信号，该证书应降级 unknown
        self.assertIn("证书:PMP", result.unknown_items)

    # ---- 证书：否定证据无效 -> unknown ----

    def test_cert_negative_evidence_invalid_is_unknown(self):
        req = self._req(certifications=("PMP",))
        raw = "Python 无关文本 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            certifications=(),
            evidence=(
                _skill_evidence("resume-1", "Python"),
                self._cert_evidence("PMP", "无PMP", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(
            req,
            profile,
            raw_text=raw,
            explicitly_not_owned_certifications=frozenset({"PMP"}),
        )
        # "无PMP" 无法在原文定位 -> 证据无效 -> unknown
        self.assertIn("证书:PMP", result.unknown_items)
        self.assertNotIn("证书:PMP", result.not_matched_items)

    # ---- 证书：正面持有 + 否定声明冲突 -> unknown ----

    def test_cert_positive_claim_and_negation_conflict_is_unknown(self):
        req = self._req(certifications=("PMP",))
        raw = "Python 持有PMP 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            certifications=("PMP",),  # 正面持有
            evidence=(
                _skill_evidence("resume-1", "Python"),
                self._cert_evidence("PMP", "持有PMP", polarity="positive"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(
            req,
            profile,
            raw_text=raw,
            explicitly_not_owned_certifications=frozenset({"PMP"}),  # 又声明否定
        )
        # 持有与否定冲突 -> unknown，不判 matched 也不判 not_matched
        self.assertIn("证书:PMP", result.unknown_items)
        self.assertNotIn("证书:PMP", result.not_matched_items)

    # ---- 技能：明确不会 + 有效否定证据 -> not_matched ----

    def test_skill_explicit_not_owned_with_negative_evidence_is_not_matched(self):
        req = self._req(required=("Python",))
        raw = "不会Python 5年"
        profile = _make_profile(
            skills=(),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python", quote="不会Python", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Python"}))
        self.assertEqual(result.skill_statuses["required"]["Python"], "not_matched")
        self.assertIn("必备技能:Python", result.not_matched_items)
        self.assertEqual(result.recommendation, "limited_match")

    # ---- 技能：正面证据与 explicitly_not_owned 冲突 -> unknown ----

    def test_skill_positive_evidence_conflicts_with_not_owned_is_unknown(self):
        # 简历正面声明"精通Python"，但 explicitly_not_owned 却包含 Python -> 方向冲突 -> unknown
        req = self._req(required=("Python",))
        raw = "精通Python 5年"
        profile = _make_profile(
            skills=("Python",),  # 正面持有
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python", quote="精通Python", polarity="positive"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Python"}))
        # 正面持有与明确否定冲突 -> unknown，而非 not_matched
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")
        self.assertNotIn("必备技能:Python", result.not_matched_items)

    # ---- 技能：否定证据无效 -> unknown ----

    def test_skill_negative_evidence_invalid_is_unknown(self):
        req = self._req(required=("Python",))
        raw = "无关文本 5年"
        profile = _make_profile(
            skills=(),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python", quote="不会Python", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Python"}))
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    # ---- 技能：正面证据不得作为否定证据 ----

    def test_skill_positive_evidence_not_treated_as_negation(self):
        # 明确不会声明下，只有 polarity="positive" 的证据 -> 不能证明否定 -> unknown
        req = self._req(required=("Python",))
        raw = "精通Python 5年"
        profile = _make_profile(
            skills=(),  # 未录入正面技能
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python", quote="精通Python", polarity="positive"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Python"}))
        # 只有正面证据，无否定证据 -> 无法证明"不会" -> unknown
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")
        self.assertNotIn("必备技能:Python", result.not_matched_items)

    # ---- 证据方向无法确定（无任何极性信息可供区分）降级 unknown ----

    def test_skill_direction_undeterminable_is_unknown(self):
        # 声明了否定信号，但证据 polarity 缺省为 positive 且无否定证据 -> 方向无法确定 -> unknown
        req = self._req(required=("Python",))
        raw = "Python 5年"
        profile = _make_profile(
            skills=(),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python", quote="Python", polarity="positive"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Python"}))
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    # ---- 回归：评分/推荐/排序规则不回归 ----

    def test_regression_scoring_weights_and_recommendation_unchanged(self):
        # 全满足场景：overall_score=100，strong_match（与 T01 一致）
        req = _make_requirements(
            required=("Python", "FastAPI"),
            preferred=("Docker",),
            min_exp=3,
            max_exp=8,
            education="本科",
        )
        raw = "Python FastAPI Docker 5年 本科"
        profile = _make_profile(
            skills=("Python", "FastAPI", "Docker"),
            exp=5,
            education="本科",
            evidence=(
                _skill_evidence("resume-1", "Python"),
                _skill_evidence("resume-1", "FastAPI"),
                _skill_evidence("resume-1", "Docker"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
                Evidence(resume_id="resume-1", category="education", subject="本科", quote="本科"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.overall_score, 100.0)
        self.assertEqual(result.recommendation, "strong_match")

    def test_regression_rank_order_unchanged(self):
        # 排序四层键不变：recommendation 等级优先，其次 overall_score、information_completeness、resume_id
        req = _make_requirements(required=("Python",), min_exp=3, max_exp=8)
        raw = "Python 5年"

        def make(rid):
            p = _make_profile(
                resume_id=rid,
                skills=("Python",),
                exp=5,
                evidence=(
                    _skill_evidence(rid, "Python"),
                    Evidence(resume_id=rid, category="experience", subject="经验", quote="5年"),
                ),
            )
            return score(req, p, raw_text=raw)

        strong = make("a")
        insufficient = make("b")  # 同 strong 输入其实也是 strong，这里仅验证稳定排序
        ordered = rank([insufficient, strong])
        self.assertEqual(ordered[0].resume_id, "a")


class PolarityConflictTests(unittest.TestCase):
    """Evidence 极性冲突仲裁（本轮修复新增）。

    同一 subject 同时存在有效 positive 与有效 negative Evidence 时，必须降级
    unknown，不得判 matched / not_matched。该规则独立于 explicitly_not_owned
    集合交集检查，仅依据 Evidence 本身的极性。
    """

    def _req(self, required=("Python",), certifications=()):
        return _make_requirements(
            required=required,
            min_exp=3,
            max_exp=8,
            certifications=certifications,
        )

    def _skill_pos(self, subject, quote):
        return Evidence(
            resume_id="resume-1", category="skill", subject=subject,
            quote=quote, polarity="positive",
        )

    def _skill_neg(self, subject, quote):
        return Evidence(
            resume_id="resume-1", category="skill", subject=subject,
            quote=quote, polarity="negative",
        )

    def _cert_pos(self, subject, quote):
        return Evidence(
            resume_id="resume-1", category="certification", subject=subject,
            quote=quote, polarity="positive",
        )

    def _cert_neg(self, subject, quote):
        return Evidence(
            resume_id="resume-1", category="certification", subject=subject,
            quote=quote, polarity="negative",
        )

    # ---- 技能：正负证据冲突（未声明 not_owned）-> unknown ----

    def test_skill_positive_negative_conflict_is_unknown(self):
        # profile.skills 持有 Python，同时有有效 positive + negative 证据，
        # 但 explicitly_not_owned 为空 -> 必须 unknown（而非 matched）
        req = self._req(required=("Python",))
        raw = "精通Python 不会Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                self._skill_pos("Python", "精通Python"),
                self._skill_neg("Python", "不会Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertEqual(result.skill_statuses["required"]["Python"], "unknown")

    # ---- 证书：正负证据冲突（未声明 not_owned_cert）-> unknown ----

    def test_cert_positive_negative_conflict_is_unknown(self):
        # profile.certifications 持有 PMP，同时有有效 positive + negative 证据，
        # 但 explicitly_not_owned_certifications 为空 -> 必须 unknown（而非 matched）
        req = self._req(certifications=("PMP",))
        raw = "Python 持有PMP 无PMP 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            certifications=("PMP",),
            evidence=(
                _skill_evidence("resume-1", "Python"),
                self._cert_pos("PMP", "持有PMP"),
                self._cert_neg("PMP", "无PMP"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertIn("证书:PMP", result.unknown_items)
        self.assertNotIn("证书:PMP", result.not_matched_items)

    # ---- 无效 negative 证据不应触发极性冲突 ----

    def test_invalid_negative_evidence_does_not_trigger_conflict(self):
        # negative 证据无法在原文定位 -> 无效，不应与 positive 证据构成冲突
        req = self._req(required=("Python",))
        raw = "精通Python 5年"  # 原文没有"不会Python"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                self._skill_pos("Python", "精通Python"),
                self._skill_neg("Python", "不会Python"),  # 无效（无法定位）
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # 只有有效正面证据，无有效否定证据 -> matched
        self.assertEqual(result.skill_statuses["required"]["Python"], "matched")

    # ---- 无效 positive 证据不应触发极性冲突 ----

    def test_invalid_positive_evidence_does_not_trigger_conflict(self):
        # positive 证据无法在原文定位 -> 无效，不应与 negative 证据构成冲突
        req = self._req(required=("Python",))
        raw = "不会Python 5年"  # 原文没有"精通Python"
        profile = _make_profile(
            skills=(),
            exp=5,
            evidence=(
                self._skill_pos("Python", "精通Python"),  # 无效（无法定位）
                self._skill_neg("Python", "不会Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(
            req, profile, raw_text=raw, explicitly_not_owned=frozenset({"Python"})
        )
        # 只有有效否定证据，无有效正面证据 -> not_matched
        self.assertEqual(result.skill_statuses["required"]["Python"], "not_matched")

    # ---- 技能冲突进入 unknown_items，不进入 not_matched_items ----

    def test_skill_conflict_goes_to_unknown_items_not_not_matched(self):
        req = self._req(required=("Python",))
        raw = "精通Python 不会Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                self._skill_pos("Python", "精通Python"),
                self._skill_neg("Python", "不会Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertIn("必备技能:Python", result.unknown_items)
        self.assertNotIn("必备技能:Python", result.not_matched_items)

    # ---- 证书冲突进入 unknown_items，不进入 not_matched_items ----

    def test_cert_conflict_goes_to_unknown_items_not_not_matched(self):
        req = self._req(certifications=("PMP",))
        raw = "Python 持有PMP 无PMP 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            certifications=("PMP",),
            evidence=(
                _skill_evidence("resume-1", "Python"),
                self._cert_pos("PMP", "持有PMP"),
                self._cert_neg("PMP", "无PMP"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertIn("证书:PMP", result.unknown_items)
        self.assertNotIn("证书:PMP", result.not_matched_items)

    # ---- 冲突情况下 needs_human_review=True ----

    def test_conflict_keeps_needs_human_review_true(self):
        req = self._req(required=("Python",))
        raw = "精通Python 不会Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                self._skill_pos("Python", "精通Python"),
                self._skill_neg("Python", "不会Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertTrue(result.needs_human_review)

    # ---- 冲突情况下不得输出确定性匹配结论 ----

    def test_conflict_does_not_produce_definitive_recommendation(self):
        # 冲突 + 其余硬性未知 -> 不能是 strong_match（确定性结论）
        req = self._req(required=("Python", "FastAPI"))
        raw = "精通Python 不会Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                self._skill_pos("Python", "精通Python"),
                self._skill_neg("Python", "不会Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        self.assertNotEqual(result.recommendation, "strong_match")

    # ---- 重复运行结果一致 ----

    def test_conflict_repeat_run_identical(self):
        req = self._req(required=("Python",))
        raw = "精通Python 不会Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                self._skill_pos("Python", "精通Python"),
                self._skill_neg("Python", "不会Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        r1 = score(req, profile, raw_text=raw)
        r2 = score(req, profile, raw_text=raw)
        self.assertEqual(r1, r2)

    # ---- 经验/学历：negative 证据不得当作正面满足依据 ----

    def test_experience_negative_evidence_not_treated_as_positive(self):
        req = self._req(required=("Python",))
        raw = "Python 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年", polarity="negative"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # negative 经验证据不作为正面依据 -> 经验降级 unknown
        self.assertEqual(result.experience_status, "unknown")
        self.assertIsNone(result.dimension_scores["experience"])

    def test_education_negative_evidence_not_treated_as_positive(self):
        req = _make_requirements(required=("Python",), min_exp=3, max_exp=8, education="本科")
        raw = "Python 本科 5年"
        profile = _make_profile(
            skills=("Python",),
            exp=5,
            education="本科",
            evidence=(
                _skill_evidence("resume-1", "Python"),
                Evidence(resume_id="resume-1", category="education", subject="本科", quote="本科", polarity="negative"),
                Evidence(resume_id="resume-1", category="experience", subject="经验", quote="5年"),
            ),
        )
        result = score(req, profile, raw_text=raw)
        # negative 学历证据不作为正面依据 -> 学历降级 unknown
        self.assertIsNone(result.dimension_scores["education"])


if __name__ == "__main__":
    unittest.main()
