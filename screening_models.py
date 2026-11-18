from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


EvidenceCategory = Literal[
    "skill", "experience", "education", "certification", "work_history", "other"
]
EvidencePolarity = Literal["positive", "negative"]
ExperienceStatus = Literal["met", "not_met", "unknown", "not_applicable"]
SkillStatus = Literal["matched", "not_matched", "unknown"]
MatchRecommendation = Literal[
    "strong_match", "potential_match", "limited_match", "insufficient_information"
]

_EVIDENCE_CATEGORIES = frozenset(
    {"skill", "experience", "education", "certification", "work_history", "other"}
)
_EVIDENCE_POLARITIES = frozenset({"positive", "negative"})
_EXPERIENCE_STATUSES = frozenset({"met", "not_met", "unknown", "not_applicable"})
_SKILL_STATUSES = frozenset({"matched", "not_matched", "unknown"})
_MATCH_RECOMMENDATIONS = frozenset(
    {"strong_match", "potential_match", "limited_match", "insufficient_information"}
)
_DIMENSION_KEYS = frozenset(
    {"required", "preferred", "experience", "education", "certification"}
)


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串。")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} 不能为空。")
    return cleaned


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _text_tuple(
    value: tuple[str, ...], field_name: str, *, unique: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} 必须是字符串元组。")
    cleaned_items = [_required_text(item, f"{field_name} 的条目") for item in value]
    if unique:
        folded = [item.casefold() for item in cleaned_items]
        if len(set(folded)) != len(folded):
            raise ValueError(f"{field_name} 不能包含重复条目。")
    return tuple(cleaned_items)


def _optional_number(
    value: float | None,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须是数值或 None。")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数值。")
    if not minimum <= number <= maximum:
        raise ValueError(f"{field_name} 必须位于 {minimum:g}–{maximum:g} 之间。")
    return number


def _non_negative_offset(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} 必须是整数或 None。")
    if value < 0:
        raise ValueError(f"{field_name} 不能为负数。")
    return value


@dataclass(frozen=True)
class ScreeningInput:
    """一次筛选的原始文本输入；原文保持不变，避免破坏证据偏移。"""

    job_description: str
    resume_texts: tuple[str, ...]
    source_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.job_description, "job_description")
        if not isinstance(self.resume_texts, tuple):
            raise TypeError("resume_texts 必须是字符串元组。")
        if not self.resume_texts:
            raise ValueError("至少需要提供一份简历。")
        for resume_text in self.resume_texts:
            _required_text(resume_text, "resume_texts 的条目")
        if not isinstance(self.source_names, tuple):
            raise TypeError("source_names 必须是字符串元组。")
        if self.source_names and len(self.source_names) != len(self.resume_texts):
            raise ValueError("source_names 为空或必须与 resume_texts 数量一致。")
        for source_name in self.source_names:
            _required_text(source_name, "source_names 的条目")


@dataclass(frozen=True)
class Evidence:
    """简历中的最小原文证据，可按页码或字符偏移复核。

    polarity 表示证据方向：
        - "positive"：简历正面声明「持有/掌握」该 subject。
        - "negative"：简历明确声明「不具备/不会」该 subject。
    默认 "positive"，用于所有正面匹配判定；否定判定必须显式使用
    "negative" 极性，避免把正面证据误当作否定证据。
    """

    resume_id: str
    category: EvidenceCategory
    subject: str
    quote: str
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    polarity: EvidencePolarity = "positive"

    def __post_init__(self) -> None:
        object.__setattr__(self, "resume_id", _required_text(self.resume_id, "resume_id"))
        if self.category not in _EVIDENCE_CATEGORIES:
            raise ValueError(f"不支持证据类别：{self.category!r}。")
        object.__setattr__(self, "subject", _required_text(self.subject, "subject"))
        _required_text(self.quote, "quote")
        if self.polarity not in _EVIDENCE_POLARITIES:
            raise ValueError(f"不支持证据极性：{self.polarity!r}。")
        if self.page_number is not None:
            if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
                raise TypeError("page_number 必须是整数或 None。")
            if self.page_number < 1:
                raise ValueError("page_number 必须从 1 开始。")
        start = _non_negative_offset(self.start_offset, "start_offset")
        end = _non_negative_offset(self.end_offset, "end_offset")
        if (start is None) != (end is None):
            raise ValueError("start_offset 和 end_offset 必须同时提供。")
        if start is not None and end <= start:
            raise ValueError("end_offset 必须大于 start_offset。")

    def matches_text(self, raw_text: str) -> bool:
        """核对引文是否存在；提供偏移时，还要求偏移切片与引文完全一致。"""
        if not isinstance(raw_text, str):
            raise TypeError("raw_text 必须是字符串。")
        if self.quote not in raw_text:
            return False
        if self.start_offset is None:
            return True
        return raw_text[self.start_offset:self.end_offset] == self.quote


@dataclass(frozen=True)
class ResumeProfile:
    """与招聘相关且可由证据复核的简历结构化信息。"""

    resume_id: str
    source_name: str
    candidate_name: str | None
    skills: tuple[str, ...]
    work_experience_years: float | None
    education: str | None
    certifications: tuple[str, ...]
    work_history: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    extraction_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "resume_id", _required_text(self.resume_id, "resume_id"))
        object.__setattr__(self, "source_name", _required_text(self.source_name, "source_name"))
        object.__setattr__(
            self, "candidate_name", _optional_text(self.candidate_name, "candidate_name")
        )
        object.__setattr__(self, "skills", _text_tuple(self.skills, "skills", unique=True))
        object.__setattr__(
            self,
            "work_experience_years",
            _optional_number(
                self.work_experience_years, "work_experience_years", minimum=0, maximum=60
            ),
        )
        object.__setattr__(self, "education", _optional_text(self.education, "education"))
        object.__setattr__(
            self, "certifications", _text_tuple(self.certifications, "certifications", unique=True)
        )
        object.__setattr__(
            self, "work_history", _text_tuple(self.work_history, "work_history")
        )
        object.__setattr__(
            self,
            "extraction_warnings",
            _text_tuple(self.extraction_warnings, "extraction_warnings"),
        )
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, Evidence) for item in self.evidence
        ):
            raise TypeError("evidence 必须是 Evidence 元组。")
        if any(item.resume_id != self.resume_id for item in self.evidence):
            raise ValueError("简历证据的 resume_id 必须与 ResumeProfile 一致。")


@dataclass(frozen=True)
class JobRequirements:
    """从 JD 提取的岗位要求；未知信息保持为空，不自行补全。"""

    job_id: str
    title: str
    required_skills: tuple[str, ...]
    preferred_skills: tuple[str, ...]
    min_experience_years: float | None
    max_experience_years: float | None
    education_requirement: str | None
    required_certifications: tuple[str, ...]
    responsibilities: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _required_text(self.job_id, "job_id"))
        object.__setattr__(self, "title", _required_text(self.title, "title"))
        required = _text_tuple(self.required_skills, "required_skills", unique=True)
        preferred = _text_tuple(self.preferred_skills, "preferred_skills", unique=True)
        object.__setattr__(self, "required_skills", required)
        object.__setattr__(self, "preferred_skills", preferred)
        required_folded = {item.casefold() for item in required}
        if any(item.casefold() in required_folded for item in preferred):
            raise ValueError("preferred_skills 不能与 required_skills 重复。")
        minimum = _optional_number(
            self.min_experience_years, "min_experience_years", minimum=0, maximum=60
        )
        maximum = _optional_number(
            self.max_experience_years, "max_experience_years", minimum=0, maximum=60
        )
        object.__setattr__(self, "min_experience_years", minimum)
        object.__setattr__(self, "max_experience_years", maximum)
        if minimum is not None and maximum is not None and maximum < minimum:
            raise ValueError("max_experience_years 不能小于 min_experience_years。")
        object.__setattr__(
            self,
            "education_requirement",
            _optional_text(self.education_requirement, "education_requirement"),
        )
        object.__setattr__(
            self,
            "required_certifications",
            _text_tuple(self.required_certifications, "required_certifications", unique=True),
        )
        object.__setattr__(
            self, "responsibilities", _text_tuple(self.responsibilities, "responsibilities")
        )


@dataclass(frozen=True)
class MatchResult:
    """单个岗位与简历的辅助匹配结果，不表示自动录用或拒绝决定。"""

    job_id: str
    resume_id: str
    matched_required_skills: tuple[str, ...]
    missing_required_skills: tuple[str, ...]
    matched_preferred_skills: tuple[str, ...]
    skill_match_score: float | None
    experience_status: ExperienceStatus
    overall_score: float | None
    recommendation: MatchRecommendation
    explanation: str
    evidence: tuple[Evidence, ...]
    missing_information: tuple[str, ...]
    needs_human_review: bool = True
    information_completeness: float | None = None
    dimension_scores: dict[str, float | None] | None = None
    skill_statuses: dict[str, dict[str, SkillStatus]] | None = None
    unknown_items: tuple[str, ...] = ()
    not_matched_items: tuple[str, ...] = ()
    pending_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _required_text(self.job_id, "job_id"))
        object.__setattr__(self, "resume_id", _required_text(self.resume_id, "resume_id"))
        matched_required = _text_tuple(
            self.matched_required_skills, "matched_required_skills", unique=True
        )
        missing_required = _text_tuple(
            self.missing_required_skills, "missing_required_skills", unique=True
        )
        object.__setattr__(self, "matched_required_skills", matched_required)
        object.__setattr__(self, "missing_required_skills", missing_required)
        object.__setattr__(
            self,
            "matched_preferred_skills",
            _text_tuple(self.matched_preferred_skills, "matched_preferred_skills", unique=True),
        )
        matched_folded = {item.casefold() for item in matched_required}
        if any(item.casefold() in matched_folded for item in missing_required):
            raise ValueError("matched_required_skills 与 missing_required_skills 不能重叠。")
        object.__setattr__(
            self,
            "skill_match_score",
            _optional_number(self.skill_match_score, "skill_match_score", minimum=0, maximum=100),
        )
        if self.experience_status not in _EXPERIENCE_STATUSES:
            raise ValueError(f"不支持经验状态：{self.experience_status!r}。")
        object.__setattr__(
            self,
            "overall_score",
            _optional_number(self.overall_score, "overall_score", minimum=0, maximum=100),
        )
        if self.recommendation not in _MATCH_RECOMMENDATIONS:
            raise ValueError(f"不支持匹配建议：{self.recommendation!r}。")
        if self.recommendation in ("strong_match", "limited_match") and self.overall_score is None:
            raise ValueError("信息不足：明确匹配结论必须有总分。")
        object.__setattr__(self, "explanation", _required_text(self.explanation, "explanation"))
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, Evidence) for item in self.evidence
        ):
            raise TypeError("evidence 必须是 Evidence 元组。")
        if any(item.resume_id != self.resume_id for item in self.evidence):
            raise ValueError("匹配证据的 resume_id 必须与 MatchResult 一致。")
        object.__setattr__(
            self,
            "missing_information",
            _text_tuple(self.missing_information, "missing_information"),
        )
        if self.needs_human_review is not True:
            raise ValueError("招聘匹配结果必须保留人工复核。")
        object.__setattr__(
            self,
            "information_completeness",
            _optional_number(
                self.information_completeness,
                "information_completeness",
                minimum=0,
                maximum=100,
            ),
        )
        object.__setattr__(
            self,
            "dimension_scores",
            self._validate_dimension_scores(self.dimension_scores),
        )
        object.__setattr__(
            self,
            "skill_statuses",
            self._validate_skill_statuses(self.skill_statuses),
        )
        object.__setattr__(
            self,
            "unknown_items",
            _text_tuple(self.unknown_items, "unknown_items"),
        )
        object.__setattr__(
            self,
            "not_matched_items",
            _text_tuple(self.not_matched_items, "not_matched_items"),
        )
        object.__setattr__(
            self,
            "pending_aliases",
            _text_tuple(self.pending_aliases, "pending_aliases"),
        )

    @staticmethod
    def _validate_dimension_scores(
        value: dict[str, float | None] | None,
    ) -> dict[str, float | None] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("dimension_scores 必须是字典或 None。")
        for key in value:
            if key not in _DIMENSION_KEYS:
                raise ValueError(f"dimension_scores 包含未知维度：{key!r}。")
        for key, score in value.items():
            if score is not None:
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise TypeError(f"dimension_scores[{key!r}] 必须是数值或 None。")
                number = float(score)
                if not math.isfinite(number) or not 0 <= number <= 100:
                    raise ValueError(f"dimension_scores[{key!r}] 必须位于 0–100 之间。")
        return dict(value)

    @staticmethod
    def _validate_skill_statuses(
        value: dict[str, dict[str, SkillStatus]] | None,
    ) -> dict[str, dict[str, SkillStatus]] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("skill_statuses 必须是字典或 None。")
        for key in value:
            if key not in {"required", "preferred"}:
                raise ValueError(f"skill_statuses 包含未知类别：{key!r}。")
        for key, statuses in value.items():
            if not isinstance(statuses, dict):
                raise TypeError(f"skill_statuses[{key!r}] 必须是字典。")
            for skill, status in statuses.items():
                if not isinstance(skill, str) or not skill.strip():
                    raise ValueError(f"skill_statuses[{key!r}] 的技能名必须是非空字符串。")
                if status not in _SKILL_STATUSES:
                    raise ValueError(
                        f"skill_statuses[{key!r}][{skill!r}] 状态无效：{status!r}。"
                    )
        return {key: dict(statuses) for key, statuses in value.items()}


def validate_match_result(
    result: MatchResult, profile: ResumeProfile, requirements: JobRequirements
) -> None:
    """执行需要同时参照匹配结果、简历和岗位要求的跨对象校验。"""
    if not isinstance(result, MatchResult) or not isinstance(profile, ResumeProfile):
        raise TypeError("result 和 profile 类型不正确。")
    if not isinstance(requirements, JobRequirements):
        raise TypeError("requirements 必须是 JobRequirements。")
    if result.resume_id != profile.resume_id:
        raise ValueError("MatchResult 与 ResumeProfile 的 resume_id 不一致。")
    if result.job_id != requirements.job_id:
        raise ValueError("MatchResult 与 JobRequirements 的 job_id 不一致。")

    def folded(items: tuple[str, ...]) -> set[str]:
        return {item.casefold() for item in items}

    required = folded(requirements.required_skills)
    matched = folded(result.matched_required_skills)
    missing = folded(result.missing_required_skills)
    if matched | missing != required:
        raise ValueError("已匹配与未匹配的必备技能必须完整覆盖岗位要求。")
    if not folded(result.matched_preferred_skills) <= folded(requirements.preferred_skills):
        raise ValueError("已匹配加分技能必须属于岗位 preferred_skills。")
    if not required and result.skill_match_score is not None:
        raise ValueError("岗位没有必备技能时，skill_match_score 必须为 None。")
    if required and result.skill_match_score is None:
        raise ValueError("岗位有必备技能时，skill_match_score 不能为 None。")
    has_experience_requirement = (
        requirements.min_experience_years is not None
        or requirements.max_experience_years is not None
    )
    if not has_experience_requirement:
        if result.experience_status != "not_applicable":
            raise ValueError("岗位无经验要求时，experience_status 必须为 not_applicable。")
    elif profile.work_experience_years is None:
        if result.experience_status != "unknown":
            raise ValueError("候选人经验信息不足时，experience_status 必须为 unknown。")
