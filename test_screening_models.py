import json
import math
import unittest
from dataclasses import asdict, replace

from screening_models import (
    Evidence,
    JobRequirements,
    MatchResult,
    ResumeProfile,
    ScreeningInput,
    validate_match_result,
)


class ScreeningModelTests(unittest.TestCase):
    def setUp(self):
        self.raw_text = "张三\nPython FastAPI Docker\n5 年经验"
        start = self.raw_text.index("Python")
        self.evidence = Evidence(
            resume_id="resume-1",
            category="skill",
            subject="Python",
            quote="Python",
            page_number=1,
            start_offset=start,
            end_offset=start + len("Python"),
        )
        self.profile = ResumeProfile(
            resume_id="resume-1",
            source_name="resume.pdf",
            candidate_name="张三",
            skills=("Python", "FastAPI", "Docker"),
            work_experience_years=5,
            education="本科",
            certifications=(),
            work_history=("五年后端开发经验",),
            evidence=(self.evidence,),
        )
        self.requirements = JobRequirements(
            job_id="job-1",
            title="后端工程师",
            required_skills=("Python", "FastAPI"),
            preferred_skills=("Docker",),
            min_experience_years=3,
            max_experience_years=8,
            education_requirement="本科",
            required_certifications=(),
            responsibilities=("开发后端服务",),
        )
        self.result = MatchResult(
            job_id="job-1",
            resume_id="resume-1",
            matched_required_skills=("Python", "FastAPI"),
            missing_required_skills=(),
            matched_preferred_skills=("Docker",),
            skill_match_score=100,
            experience_status="met",
            overall_score=92.5,
            recommendation="strong_match",
            explanation="必备技能和经验均有简历信息支持。",
            evidence=(self.evidence,),
            missing_information=(),
        )

    def test_screening_input_preserves_raw_text_and_accepts_valid_names(self):
        raw = "\n  原始简历文本  \n"
        screening = ScreeningInput("  后端岗位  ", (raw,), ("resume.pdf",))
        self.assertEqual(screening.resume_texts[0], raw)
        self.assertEqual(screening.job_description, "  后端岗位  ")

    def test_screening_input_rejects_empty_or_misaligned_values(self):
        invalid_cases = (
            {"job_description": " ", "resume_texts": ("简历",)},
            {"job_description": "岗位", "resume_texts": ()},
            {"job_description": "岗位", "resume_texts": "简历"},
            {"job_description": "岗位", "resume_texts": ("简历", " ")},
            {"job_description": "岗位", "resume_texts": ("简历",), "source_names": ("a", "b")},
            {"job_description": "岗位", "resume_texts": ("简历",), "source_names": (" ",)},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                ScreeningInput(**kwargs)

    def test_evidence_verifies_quote_and_exact_offsets(self):
        self.assertTrue(self.evidence.matches_text(self.raw_text))
        self.assertFalse(self.evidence.matches_text("没有相关技能"))
        self.assertFalse(replace(self.evidence, quote="FastAPI").matches_text(self.raw_text))
        without_offsets = replace(self.evidence, start_offset=None, end_offset=None)
        self.assertTrue(without_offsets.matches_text(self.raw_text))

    def test_evidence_rejects_invalid_category_and_locators(self):
        invalid_cases = (
            {"category": "invalid"},
            {"quote": " "},
            {"page_number": 0},
            {"page_number": True},
            {"start_offset": -1, "end_offset": 2},
            {"start_offset": 1, "end_offset": None},
            {"start_offset": 2, "end_offset": 2},
            {"polarity": "neutral"},
            {"polarity": ""},
            {"polarity": "Positive"},  # 大小写敏感，非法
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                replace(self.evidence, **changes)

    def test_evidence_accepts_valid_polarities(self):
        # positive / negative 均为合法极性，默认 positive
        self.assertEqual(self.evidence.polarity, "positive")
        negative = replace(self.evidence, polarity="negative")
        self.assertEqual(negative.polarity, "negative")

    def test_resume_profile_normalizes_display_fields(self):
        profile = replace(self.profile, source_name="  resume.pdf  ", skills=(" Python ",))
        self.assertEqual(profile.source_name, "resume.pdf")
        self.assertEqual(profile.skills, ("Python",))
        self.assertIsInstance(profile.work_experience_years, float)

    def test_resume_profile_rejects_duplicate_skills_and_foreign_evidence(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            replace(self.profile, skills=("Python", "python"))
        foreign = replace(self.evidence, resume_id="resume-2")
        with self.assertRaisesRegex(ValueError, "resume_id"):
            replace(self.profile, evidence=(foreign,))

    def test_resume_profile_rejects_invalid_experience_numbers(self):
        for value in (True, -1, 61, math.nan, math.inf, "5"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                replace(self.profile, work_experience_years=value)

    def test_job_requirements_allow_unknown_fields_without_inference(self):
        requirements = JobRequirements(
            job_id="job-empty",
            title="岗位",
            required_skills=(),
            preferred_skills=(),
            min_experience_years=None,
            max_experience_years=None,
            education_requirement=None,
            required_certifications=(),
            responsibilities=(),
        )
        self.assertIsNone(requirements.min_experience_years)
        self.assertIsNone(requirements.max_experience_years)

    def test_job_requirements_reject_invalid_range_and_skill_overlap(self):
        with self.assertRaisesRegex(ValueError, "不能小于"):
            replace(self.requirements, min_experience_years=8, max_experience_years=3)
        with self.assertRaisesRegex(ValueError, "不能与"):
            replace(self.requirements, preferred_skills=("python",))
        with self.assertRaises((TypeError, ValueError)):
            replace(self.requirements, min_experience_years=math.nan)

    def test_valid_match_result_passes_cross_object_validation(self):
        validate_match_result(self.result, self.profile, self.requirements)

    def test_match_result_is_json_serializable(self):
        payload = json.dumps(asdict(self.result), ensure_ascii=False)
        self.assertIn("strong_match", payload)
        self.assertIn("Python", payload)
        self.assertTrue(payload)

    def test_match_result_rejects_invalid_scores_and_overlapping_skills(self):
        for field in ("skill_match_score", "overall_score"):
            for value in (-1, 101, math.nan, True):
                with self.subTest(field=field, value=value), self.assertRaises((TypeError, ValueError)):
                    replace(self.result, **{field: value})
        with self.assertRaisesRegex(ValueError, "不能重叠"):
            replace(self.result, missing_required_skills=("Python",))

    def test_match_result_requires_consistent_recommendation_and_human_review(self):
        with self.assertRaisesRegex(ValueError, "信息不足"):
            replace(self.result, overall_score=None)
        # v3.4 放宽：insufficient_information 时 overall_score 允许为 None（无可靠维度）
        replace(self.result, recommendation="insufficient_information", overall_score=None)
        with self.assertRaisesRegex(ValueError, "人工复核"):
            replace(self.result, needs_human_review=False)

    def test_cross_validation_rejects_mismatched_identifiers(self):
        with self.assertRaisesRegex(ValueError, "resume_id"):
            validate_match_result(
                replace(self.result, resume_id="resume-2", evidence=()),
                self.profile,
                self.requirements,
            )
        with self.assertRaisesRegex(ValueError, "job_id"):
            validate_match_result(
                replace(self.result, job_id="job-2"), self.profile, self.requirements
            )

    def test_cross_validation_requires_complete_required_skill_coverage(self):
        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            validate_match_result(
                replace(self.result, matched_required_skills=("Python",)),
                self.profile,
                self.requirements,
            )
        with self.assertRaisesRegex(ValueError, "preferred_skills"):
            validate_match_result(
                replace(self.result, matched_preferred_skills=("Kubernetes",)),
                self.profile,
                self.requirements,
            )

    def test_cross_validation_enforces_score_availability(self):
        no_skills = replace(
            self.requirements, required_skills=(), preferred_skills=(), min_experience_years=3
        )
        no_skill_result = replace(
            self.result,
            matched_required_skills=(),
            matched_preferred_skills=(),
            skill_match_score=None,
        )
        validate_match_result(no_skill_result, self.profile, no_skills)
        with self.assertRaisesRegex(ValueError, "必须为 None"):
            validate_match_result(
                replace(no_skill_result, skill_match_score=100), self.profile, no_skills
            )
        with self.assertRaisesRegex(ValueError, "不能为 None"):
            validate_match_result(
                replace(self.result, skill_match_score=None), self.profile, self.requirements
            )

    def test_cross_validation_marks_missing_experience_as_unknown(self):
        unknown_profile = replace(self.profile, work_experience_years=None)
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_match_result(self.result, unknown_profile, self.requirements)
        unknown_result = replace(self.result, experience_status="unknown")
        validate_match_result(unknown_result, unknown_profile, self.requirements)


if __name__ == "__main__":
    unittest.main()
