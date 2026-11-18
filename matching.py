"""招聘筛选 Matching Engine —— 确定性评分引擎（v3.4）。

本模块是纯函数实现：相同输入必然得到相同输出，不调用 LLM、不访问网络、
不使用随机数或时间。LLM 只负责上游的结构化抽取（产出 ResumeProfile /
JobRequirements / Evidence），本模块只消费这些已经固化的数据并计算分数。

评分维度与权重（固定）：
    - 必备技能 required     50%
    - 加分技能 preferred    10%
    - 工作经验 experience   20%
    - 学历 education         10%
    - 证书 certification     10%
    合计 100%。

每个维度的状态为四态之一：met / not_met / unknown / not_applicable。
unknown 从评分分母剔除，绝不按 0 分处理；not_applicable 表示岗位未提要求。
"""

from __future__ import annotations

import math
from typing import Iterable

from screening_models import (
    Evidence,
    EvidencePolarity,
    JobRequirements,
    MatchRecommendation,
    MatchResult,
    ResumeProfile,
    SkillStatus,
)

# 固定权重（合计 1.0）
_WEIGHTS: dict[str, float] = {
    "required": 0.50,
    "preferred": 0.10,
    "experience": 0.20,
    "education": 0.10,
    "certification": 0.10,
}

# 候选人排序等级（优 -> 劣）。注意：此顺序仅用于排序，不改变 recommendation 判定。
_RECOMMENDATION_RANK: dict[MatchRecommendation, int] = {
    "strong_match": 0,
    "potential_match": 1,
    "insufficient_information": 2,
    "limited_match": 3,
}

# 学历的等价层级（用于判定"满足/不满足"，仅做最保守的显式等价）。
# 数字越小层级越高；学历比对仅用于明确可比的情形，无法判定时返回 unknown。
_EDUCATION_LEVEL: dict[str, int] = {
    "博士": 0,
    "硕士": 1,
    "研究生": 1,
    "本科": 2,
    "大专": 3,
    "专科": 3,
    "高中": 4,
    "中专": 4,
}


def _round1(value: float) -> float:
    """统一舍入到 1 位小数，保证可复现。"""
    return round(value, 1)


def _fold_skills(skills: Iterable[str]) -> set[str]:
    """技能归一化（大小写折叠 + 去首尾空白），不涉及别名。"""
    return {skill.strip().casefold() for skill in skills if skill.strip()}


def _apply_aliases(skills: set[str], aliases: dict[str, str] | None) -> set[str]:
    """将已确认的别名映射到规范技能名（仅 L1/L3 人工确认别名）。"""
    if not aliases:
        return set(skills)
    result: set[str] = set()
    for skill in skills:
        target = aliases.get(skill, aliases.get(skill.casefold()))
        result.add(target.casefold() if target else skill)
    return result


def _valid_evidence(evidence: Evidence, raw_text: str | None) -> bool:
    """判断一条 Evidence 是否有效（§8）。

    要求 quote 能在原文中定位；若提供偏移，则偏移切片必须与引文完全一致。

    当 raw_text 为 None 时，无法做原文定位校验，按 v3.4 规定**不得视为可信**，
    因此返回 False（该证据无效，对应判定降级为 unknown）。
    """
    if raw_text is None:
        return False
    try:
        return evidence.matches_text(raw_text)
    except (TypeError, ValueError):
        return False


def _evidence_supports(
    evidence: tuple[Evidence, ...],
    category: str,
    raw_text: str | None,
    subject_key: str | None = None,
    polarity: EvidencePolarity | None = None,
) -> bool:
    """是否存在一条类别匹配且有效的证据。

    当 subject_key 提供时，还要求 subject 与被判定技能等价（casefold 后相等）。
    当 polarity 提供时，还要求证据方向与之相符（positive/negative）。
    """
    for item in evidence:
        if item.category != category:
            continue
        if polarity is not None and item.polarity != polarity:
            continue
        if not _valid_evidence(item, raw_text):
            continue
        if subject_key is not None:
            if item.subject.strip().casefold() != subject_key:
                continue
        return True
    return False


def _has_polarity_conflict(
    evidence: tuple[Evidence, ...],
    category: str,
    raw_text: str | None,
    subject_key: str,
) -> bool:
    """检测同一 subject 是否同时存在有效正面与有效否定证据。

    这是「证据方向冲突」仲裁：当同一 subject 既有能定位的正面证据、
    又有能定位的否定证据时，方向无法确定，必须降级 unknown。
    该规则独立于 explicitly_not_owned / explicitly_not_owned_certifications
    集合交集检查，仅依据 Evidence 本身的极性。无效证据不参与判定。
    """
    has_positive = _evidence_supports(
        evidence, category, raw_text, subject_key=subject_key, polarity="positive"
    )
    has_negative = _evidence_supports(
        evidence, category, raw_text, subject_key=subject_key, polarity="negative"
    )
    return has_positive and has_negative


def _classify_skill(
    skill: str,
    profile_skills: set[str],
    evidence: tuple[Evidence, ...],
    raw_text: str | None,
    explicitly_not_owned: set[str],
) -> SkillStatus:
    """判定单个技能的三态：matched / not_matched / unknown。

    - matched：简历技能集中存在等价技能，且有有效正面 Evidence 支持。
    - not_matched：简历明确声明不具备（explicitly_not_owned），且存在
      有效**否定**（polarity="negative"）Evidence 支持。
    - unknown：简历未提及；或提及但无有效证据；或正面/否定证据方向冲突。
    """
    key = skill.strip().casefold()
    claimed = key in profile_skills
    negated = key in explicitly_not_owned

    # 证据方向冲突仲裁（独立于集合交集检查）：同一技能同时存在
    # 有效正面与有效否定证据 -> 方向无法确定 -> unknown。
    if _has_polarity_conflict(evidence, "skill", raw_text, key):
        return "unknown"

    # 正面持有与明确否定同时存在 -> 证据方向冲突 -> 无法确定 -> unknown
    if claimed and negated:
        return "unknown"

    if claimed:
        # 声称掌握：必须有效正面证据，否则降级 unknown
        if _evidence_supports(evidence, "skill", raw_text, subject_key=key, polarity="positive"):
            return "matched"
        return "unknown"

    if negated:
        # 明确不会：必须有效否定证据，否则降级 unknown
        if _evidence_supports(evidence, "skill", raw_text, subject_key=key, polarity="negative"):
            return "not_matched"
        return "unknown"

    return "unknown"


def _skill_score(matched: int, not_matched: int) -> float | None:
    """按 matched / (matched + not_matched) * 100 计算技能分。

    unknown 被剔除出分母；当分母为 0（全部 unknown）时返回 None。
    """
    denominator = matched + not_matched
    if denominator == 0:
        return None
    return _round1(100.0 * matched / denominator)


def _experience_score(
    requirements: JobRequirements, profile: ResumeProfile, evidence: tuple[Evidence, ...], raw_text: str | None
) -> tuple[str, float | None]:
    """计算经验维度得分，返回 (experience_status, experience_score)。"""
    has_requirement = (
        requirements.min_experience_years is not None
        or requirements.max_experience_years is not None
    )
    if not has_requirement:
        return "not_applicable", None
    if profile.work_experience_years is None:
        return "unknown", None
    # 经验年限必须有有效**正面** Evidence 支撑，否则不可信 -> unknown。
    # 否定（negative）证据表示"无经验"，不得作为满足正面要求的依据。
    if not _evidence_supports(evidence, "experience", raw_text, polarity="positive"):
        return "unknown", None

    exp = profile.work_experience_years
    minimum = requirements.min_experience_years
    maximum = requirements.max_experience_years

    # 负数经验已在数据类层拒绝；此处防御性二次检查。
    if exp < 0:
        raise ValueError("工作年限不能为负数。")

    if minimum is not None and exp < minimum:
        # 线性计分：100 * exp / min（min > 0 时）
        if minimum > 0:
            return "not_met", _round1(100.0 * exp / minimum)
        return "not_met", 0.0
    if maximum is not None and exp > maximum:
        # 超上限视为 met，不扣分
        return "met", 100.0
    return "met", 100.0


def _education_score(
    requirements: JobRequirements, profile: ResumeProfile, evidence: tuple[Evidence, ...], raw_text: str | None
) -> tuple[str, float | None]:
    """计算学历维度得分，返回 (education_status, education_score)。"""
    requirement = requirements.education_requirement
    if requirement is None:
        return "not_applicable", None
    if profile.education is None:
        return "unknown", None
    # 学历必须有有效**正面** Evidence 支撑，否则不可信 -> unknown。
    # 否定（negative）证据表示"不具备该学历"，不得作为满足正面要求的依据。
    if not _evidence_supports(evidence, "education", raw_text, polarity="positive"):
        return "unknown", None

    req_level = _EDUCATION_LEVEL.get(requirement.strip())
    cand_level = _EDUCATION_LEVEL.get(profile.education.strip())
    if req_level is None or cand_level is None:
        # 无法做确定性层级比较，视为未知（不自行推断）
        return "unknown", None
    if cand_level <= req_level:
        return "met", 100.0
    return "not_met", 0.0


def _certification_score(
    requirements: JobRequirements,
    profile: ResumeProfile,
    evidence: tuple[Evidence, ...],
    raw_text: str | None,
    explicitly_not_owned_certifications: set[str],
) -> tuple[str, float | None]:
    """计算证书维度得分，逐证书判定后聚合。

    返回 (certification_status, certification_score)。

    证书判定规则（修复过度推断）：
        - 简历持有该证书（profile.certifications）且有有效正面证据 -> matched。
        - 简历明确不具备（explicitly_not_owned_certifications）且有有效否定证据
          -> not_matched。
        - 简历未提及该证书 -> unknown（绝不判 not_matched）。
        - 声称持有但证据无效 -> unknown；明确否定但证据无效 -> unknown。
    """
    required = requirements.required_certifications
    if not required:
        return "not_applicable", None

    candidate_certs = _fold_skills(profile.certifications)
    not_owned_certs = _fold_skills(explicitly_not_owned_certifications)
    matched = 0
    not_matched = 0
    for cert in required:
        status = _cert_status(
            cert, profile, evidence, raw_text, not_owned_certs
        )
        if status == "matched":
            matched += 1
        elif status == "not_matched":
            not_matched += 1
    score = _skill_score(matched, not_matched)
    total = len(required)
    if not_matched > 0:
        # 存在明确不具备的证书 -> not_met
        return "not_met", score
    if matched == total:
        # 所有必需证书均已确认匹配 -> met
        return "met", score
    if matched == 0:
        # 全部 unknown -> unknown，无分数
        return "unknown", None
    # 部分 matched、部分 unknown（无 not_matched）-> 信息不足，但保留已知分数
    return "unknown", score


def score(
    requirements: JobRequirements,
    profile: ResumeProfile,
    *,
    aliases: dict[str, str] | None = None,
    pending_aliases: tuple[str, ...] = (),
    raw_text: str | None = None,
    explicitly_not_owned: frozenset[str] = frozenset(),
    explicitly_not_owned_certifications: frozenset[str] = frozenset(),
) -> MatchResult:
    """对单个简历与岗位执行确定性评分。

    参数：
        requirements       岗位要求（JobRequirements）。
        profile            简历结构化信息（ResumeProfile）。
        aliases            已人工确认的技能别名映射（L1/L3）。
        pending_aliases    仅 LLM 提出的候选别名，不参与评分。
        raw_text           简历原文，用于校验 Evidence 可定位；为 None 时跳过定位校验。
        explicitly_not_owned 简历明确声明"不会/不具备"的技能集合（由抽取阶段提供）。
        explicitly_not_owned_certifications 简历明确声明"不具备"的证书集合（由抽取阶段提供）。

    返回 MatchResult，其中 overall_score 可能为 None（无任何可靠评分维度）。
    pending_aliases 只被原样记录，绝不参与数值计算。
    """
    # 候选别名不参与评分，仅透传记录（不影响任何数值计算）。

    required_skills = requirements.required_skills
    preferred_skills = requirements.preferred_skills

    # 归一化简历技能集（应用已确认别名）
    profile_skills_raw = _fold_skills(profile.skills)
    profile_skills = _apply_aliases(profile_skills_raw, aliases)
    not_owned = _apply_aliases(_fold_skills(explicitly_not_owned), aliases)
    not_owned_certs = _fold_skills(explicitly_not_owned_certifications)

    evidence = profile.evidence

    # 1) 逐技能判定三态，构建嵌套 skill_statuses
    skill_statuses: dict[str, dict[str, SkillStatus]] = {"required": {}, "preferred": {}}
    for skill in required_skills:
        skill_statuses["required"][skill] = _classify_skill(
            skill, profile_skills, evidence, raw_text, not_owned
        )
    for skill in preferred_skills:
        skill_statuses["preferred"][skill] = _classify_skill(
            skill, profile_skills, evidence, raw_text, not_owned
        )

    # 2) 必备技能分
    matched_required = [
        s for s in required_skills if skill_statuses["required"][s] == "matched"
    ]
    not_matched_required = [
        s for s in required_skills if skill_statuses["required"][s] == "not_matched"
    ]
    unknown_required = [
        s for s in required_skills if skill_statuses["required"][s] == "unknown"
    ]
    required_score = _skill_score(len(matched_required), len(not_matched_required))

    # 3) 加分技能分
    matched_preferred = [
        s for s in preferred_skills if skill_statuses["preferred"][s] == "matched"
    ]
    not_matched_preferred = [
        s for s in preferred_skills if skill_statuses["preferred"][s] == "not_matched"
    ]
    if not preferred_skills:
        preferred_score: float | None = None
    else:
        preferred_score = _skill_score(len(matched_preferred), len(not_matched_preferred))

    # 4) 经验 / 学历 / 证书
    experience_status, experience_score = _experience_score(
        requirements, profile, evidence, raw_text
    )
    education_status, education_score = _education_score(
        requirements, profile, evidence, raw_text
    )
    certification_status, certification_score = _certification_score(
        requirements, profile, evidence, raw_text, not_owned_certs
    )

    dimension_scores: dict[str, float | None] = {
        "required": required_score,
        "preferred": preferred_score,
        "experience": experience_score,
        "education": education_score,
        "certification": certification_score,
    }

    # 5) overall_score：仅累加有效维度，权重重新归一化
    weighted_sum = 0.0
    weight_sum = 0.0
    for dim, dim_score in dimension_scores.items():
        if dim_score is not None:
            weighted_sum += dim_score * _WEIGHTS[dim]
            weight_sum += _WEIGHTS[dim]
    if weight_sum > 0:
        overall_score = _round1(weighted_sum / weight_sum)
        overall_score = min(100.0, max(0.0, overall_score))
    else:
        overall_score = None

    # 6) information_completeness（独立于总分）
    #    应判定项目 = 每个必备/加分技能 + 经验(若有) + 学历(若有) + 每个必需证书
    total_applicable = 0
    reliably_determined = 0
    for statuses in (skill_statuses["required"].values(), skill_statuses["preferred"].values()):
        for status in statuses:
            total_applicable += 1
            if status in ("matched", "not_matched"):
                reliably_determined += 1
    if requirements.min_experience_years is not None or requirements.max_experience_years is not None:
        total_applicable += 1
        if experience_status in ("met", "not_met"):
            reliably_determined += 1
    if requirements.education_requirement is not None:
        total_applicable += 1
        if education_status in ("met", "not_met"):
            reliably_determined += 1
    total_applicable += len(requirements.required_certifications)
    # 证书逐项：matched/not_matched 计为可靠判定
    for cert in requirements.required_certifications:
        cert_status = _cert_status(cert, profile, evidence, raw_text, not_owned_certs)
        if cert_status in ("matched", "not_matched"):
            reliably_determined += 1
    if total_applicable > 0:
        information_completeness = _round1(100.0 * reliably_determined / total_applicable)
    else:
        information_completeness = None

    # 7) 收集 unknown_items / not_matched_items
    unknown_items: list[str] = []
    not_matched_items: list[str] = []
    for category, statuses in skill_statuses.items():
        prefix = "必备技能" if category == "required" else "加分技能"
        for skill, status in statuses.items():
            if status == "unknown":
                unknown_items.append(f"{prefix}:{skill}")
            elif status == "not_matched":
                not_matched_items.append(f"{prefix}:{skill}")
    if experience_status == "unknown":
        unknown_items.append("经验")
    elif experience_status == "not_met":
        not_matched_items.append("经验")
    if education_status == "unknown":
        unknown_items.append("学历")
    elif education_status == "not_met":
        not_matched_items.append("学历")
    for cert in requirements.required_certifications:
        cert_status = _cert_status(cert, profile, evidence, raw_text, not_owned_certs)
        if cert_status == "unknown":
            unknown_items.append(f"证书:{cert}")
        elif cert_status == "not_matched":
            not_matched_items.append(f"证书:{cert}")

    # 8) recommendation 判定（优先级从高到低）
    recommendation = _recommendation(
        required_skills,
        preferred_skills,
        skill_statuses,
        experience_status,
        education_status,
        certification_status,
    )

    # 9) 匹配/缺失必备技能、匹配加分技能（保持与 MatchResult 原有字段一致）
    matched_required_skills = tuple(matched_required)
    missing_required_skills = tuple(not_matched_required)
    matched_preferred_skills = tuple(matched_preferred)

    # 10) explanation 与缺失信息
    explanation = _build_explanation(
        recommendation, required_score, overall_score, information_completeness
    )
    missing_information = tuple(unknown_items)

    # 11) 收集支持结论的 Evidence（matched/met 项的证据）
    result_evidence = _collect_evidence(evidence, raw_text)

    return MatchResult(
        job_id=requirements.job_id,
        resume_id=profile.resume_id,
        matched_required_skills=matched_required_skills,
        missing_required_skills=missing_required_skills,
        matched_preferred_skills=matched_preferred_skills,
        skill_match_score=required_score,
        experience_status=experience_status,
        overall_score=overall_score,
        recommendation=recommendation,
        explanation=explanation,
        evidence=result_evidence,
        missing_information=missing_information,
        needs_human_review=True,
        information_completeness=information_completeness,
        dimension_scores=dimension_scores,
        skill_statuses=skill_statuses,
        unknown_items=tuple(unknown_items),
        not_matched_items=tuple(not_matched_items),
        pending_aliases=pending_aliases,
    )


def _cert_status(
    cert: str,
    profile: ResumeProfile,
    evidence: tuple[Evidence, ...],
    raw_text: str | None,
    explicitly_not_owned_certifications: set[str],
) -> SkillStatus:
    """证书的单证判定，返回 matched/not_matched/unknown。

    - matched：简历持有该证书，且存在有效正面 Evidence。
    - not_matched：简历明确不具备（explicitly_not_owned_certifications），
      且存在有效否定（polarity="negative"）Evidence。
    - unknown：未提及；或正面/否定证据缺失或方向冲突。
    """
    key = cert.strip().casefold()
    claimed = key in _fold_skills(profile.certifications)
    negated = key in explicitly_not_owned_certifications

    # 证据方向冲突仲裁（独立于集合交集检查）：同一证书同时存在
    # 有效正面与有效否定证据 -> 方向无法确定 -> unknown。
    if _has_polarity_conflict(evidence, "certification", raw_text, key):
        return "unknown"

    # 持有与明确否定同时出现 -> 方向冲突 -> unknown
    if claimed and negated:
        return "unknown"

    if claimed:
        # 声称持有：必须有效正面证据
        if _evidence_supports(evidence, "certification", raw_text, subject_key=key, polarity="positive"):
            return "matched"
        return "unknown"

    if negated:
        # 明确不具备：必须有效否定证据
        if _evidence_supports(evidence, "certification", raw_text, subject_key=key, polarity="negative"):
            return "not_matched"
        return "unknown"

    return "unknown"


def _recommendation(
    required_skills: tuple[str, ...],
    preferred_skills: tuple[str, ...],
    skill_statuses: dict[str, dict[str, SkillStatus]],
    experience_status: str,
    education_status: str,
    certification_status: str,
) -> MatchRecommendation:
    """确定性 recommendation 判定，优先级从高到低，命中即返回。

    硬性 not_met 优先于硬性 unknown；加分技能 unknown 不触发 insufficient。
    """
    # 硬性 not_met 判定
    has_hard_not_met = (
        any(s == "not_matched" for s in skill_statuses["required"].values())
        or experience_status == "not_met"
        or education_status == "not_met"
        or certification_status == "not_met"
    )
    if has_hard_not_met:
        return "limited_match"

    # 硬性 unknown 判定（必备技能 / 经验 / 学历 / 证书）
    has_hard_unknown = (
        any(s == "unknown" for s in skill_statuses["required"].values())
        or experience_status == "unknown"
        or education_status == "unknown"
        or certification_status == "unknown"
    )
    if has_hard_unknown:
        return "insufficient_information"

    # 软性门槛：加分技能（若岗位提出）须全部 matched
    preferred_statuses = skill_statuses["preferred"]
    soft_gate_passes = (
        not preferred_statuses
        or all(status == "matched" for status in preferred_statuses.values())
    )

    # 岗位没有任何要求 -> potential_match（不评 strong_match）
    no_requirements = (
        not required_skills
        and not preferred_skills
        and experience_status == "not_applicable"
        and education_status == "not_applicable"
        and certification_status == "not_applicable"
    )
    if no_requirements:
        return "potential_match"

    if soft_gate_passes:
        return "strong_match"
    return "potential_match"


def _build_explanation(
    recommendation: MatchRecommendation,
    required_score: float | None,
    overall_score: float | None,
    information_completeness: float | None,
) -> str:
    """生成匹配解释文字（确定性，非 LLM）。"""
    parts = []
    if required_score is not None:
        parts.append(f"必备技能匹配分 {required_score}")
    if overall_score is not None:
        parts.append(f"综合分 {overall_score}")
    if information_completeness is not None:
        parts.append(f"信息完整度 {information_completeness}")
    label = {
        "strong_match": "强匹配",
        "potential_match": "潜在匹配",
        "limited_match": "有限匹配",
        "insufficient_information": "信息不足",
    }[recommendation]
    base = f"结论：{label}。"
    if parts:
        base += " " + "；".join(parts) + "。"
    return base


def _collect_evidence(
    evidence: tuple[Evidence, ...], raw_text: str | None
) -> tuple[Evidence, ...]:
    """收集所有有效证据（用于 MatchResult.evidence）。"""
    return tuple(item for item in evidence if _valid_evidence(item, raw_text))


# ---- 排序 ----

def rank(results: Iterable[MatchResult]) -> list[MatchResult]:
    """对候选人的 MatchResult 进行稳定排序（四层键）。

    1. recommendation 等级（strong > potential > insufficient > limited）
    2. overall_score 降序（None 视为最低）
    3. information_completeness 降序（None 视为最低）
    4. resume_id 升序（字典序，最终稳定键）

    返回排序后的新列表，不修改输入。
    """
    def sort_key(result: MatchResult):
        rec_rank = _RECOMMENDATION_RANK[result.recommendation]
        overall = result.overall_score if result.overall_score is not None else -math.inf
        completeness = (
            result.information_completeness
            if result.information_completeness is not None
            else -math.inf
        )
        resume_id = result.resume_id
        return (rec_rank, -overall, -completeness, resume_id)

    return sorted(results, key=sort_key)
