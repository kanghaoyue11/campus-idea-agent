"""LLM 结构化抽取（招聘筛选流水线第三阶段）。

从简历原文抽取候选人的结构化信息（技能 / 经验 / 学历 / 证书 / 明确不会项）
以及对应的 Evidence。本模块复用 core.LLMClient 与 core.parse_json，
不复制、不改写 Matching Engine 的评分逻辑。

安全与合规约束：
    - quote 必须能在一份简历原文中定位，无法定位的 Evidence 一律丢弃（降级）。
    - polarity 只能是 positive / negative，其余值丢弃或报错。
    - 不确定时输出 unknown / 空，不得强行判断。
    - 不得把"未提及"当成"明确不具备"。
    - 不抽取年龄、性别、民族、婚育、照片、住址等敏感属性用于评分。
    - 抽取结果必须经过 screening_models 数据模型校验，校验失败按 fallback 处理。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from screening_models import Evidence, ResumeProfile

# 允许的 Evidence category（与 screening_models.EvidenceCategory 对齐）。
_VALID_CATEGORIES = frozenset(
    {"skill", "experience", "education", "certification", "work_history", "other"}
)
_VALID_POLARITIES = frozenset({"positive", "negative"})

# 敏感属性关键词：抽取 Prompt 明确禁止，抽取结果若混入则直接剔除。
# 多字词用于子串匹配（如"性别"命中"性别不限"）；单字词（男/女）仅精确匹配，
# 避免误伤正常技能名。
_SENSITIVE_SUBJECT_KEYWORDS = (
    "年龄", "性别", "民族", "婚", "育", "照片", "住址", "籍贯", "身份证",
    "身高", "体重", "政治面貌", "宗教信仰", "age", "gender", "marital",
)
_SENSITIVE_EXACT_KEYWORDS = frozenset({"男", "女"})


@dataclass(frozen=True)
class ExtractionSpec:
    """一次结构化抽取的输入快照（不可变，供并行/重试复用）。"""
    resume_id: str
    source_name: str
    raw_text: str


@dataclass
class ExtractionOutcome:
    """单份简历的抽取结果，含成功标记与警告。"""
    resume_id: str
    profile: ResumeProfile | None
    ok: bool
    error: str | None = None
    warnings: tuple[str, ...] = ()
    explicit_not_owned_skills: frozenset[str] = frozenset()
    explicit_not_owned_certifications: frozenset[str] = frozenset()
    # 实际调用 client.complete() 的次数（含初次请求与重试），
    # 与 Scheduler 的 RoleResult.attempts（任务级尝试次数）是不同指标。
    llm_request_attempts: int = 0


def build_extraction_prompt(spec: ExtractionSpec) -> tuple[str, str]:
    """构造结构化抽取的 system / user 提示词。

    返回 (system, user)。system 强调只从原文抽取、quote 必须逐字来自原文、
    polarity 语义、敏感信息禁止；user 提供简历原文。
    """
    system = (
        "你是简历结构化抽取 Agent。你的任务是把简历原文转换为严格 JSON，"
        "只做信息抽取，不做任何评分、判断是否录用，也不推断原文没有的信息。\n\n"
        "必须遵守：\n"
        "1. 每个证据（evidence）的 quote 必须逐字来自简历原文，禁止改写、翻译、补全。\n"
        "2. polarity 只能是 \"positive\"（简历明确声明掌握/持有）或 \"negative\"（简历明确声明不会/不具备/没有）。"
        "不确定时不要给 negative，直接不输出该证据。\n"
        "3. 简历未提及的技能、证书，一律不要列入，也绝不能标记为 negative。\n"
        "4. 禁止抽取或使用年龄、性别、民族、婚育、照片、住址、身份证等敏感属性。\n"
        "5. 工作年限无法从原文确定时，work_experience_years 用 null。\n"
        "6. 只输出 JSON，不要输出解释文字或 Markdown 代码块。\n"
        "7. skills 中的每一项都必须有一条 category=\"skill\" 的正面证据，"
        "且该证据的 subject 必须与该技能名称完全一致；找不到逐字引用就不要列入 skills。\n"
        "8. certifications 中的每一项都必须有一条 category=\"certification\" 的正面证据，"
        "且 subject 必须与证书名称完全一致；work_experience_years 非 null 时必须提供"
        "category=\"experience\"、subject=\"经验\" 的正面证据；education 非 null 时必须提供"
        "category=\"education\" 的正面证据。\n"
        "9. skills 和 evidence.subject 使用具体、规范的技能名（例如 Java、MySQL、Docker、Redis），"
        "不要使用“数据库”“缓存”“工程化”这类分类词作为技能名或证据 subject。\n\n"
        "输出 JSON 结构（字段名固定）：\n"
        '{"skills": ["技能"], "work_experience_years": 数字或null, '
        '"education": "学历或null", "certifications": ["证书"], '
        '"explicitly_not_owned_skills": ["明确不会的技能"], '
        '"explicitly_not_owned_certifications": ["明确没有的证书"], '
        '"evidence": [{"category": "skill/experience/education/certification", '
        '"subject": "对象名", "quote": "原文逐字引用", "polarity": "positive/negative"}]}'
    )
    user = f"简历原文：\n{spec.raw_text}"
    return system, user


def _validate_quote(quote: str, raw_text: str) -> bool:
    """quote 必须能在原文中逐字定位。"""
    return bool(quote) and quote in raw_text


def _clean_evidence_items(
    items: list[dict[str, Any]],
    resume_id: str,
    raw_text: str,
) -> tuple[list[Evidence], list[str]]:
    """清洗 LLM 返回的 evidence 列表，返回 (有效 Evidence 列表, 丢弃原因)。

    丢弃规则：
        - category 不在允许集合；
        - polarity 不在 {positive, negative}；
        - subject 为空；
        - quote 为空或无法在原文定位；
        - subject 命中敏感属性关键词。
    """
    valid: list[Evidence] = []
    dropped: list[str] = []
    for idx, item in enumerate(items or []):
        if not isinstance(item, dict):
            dropped.append(f"evidence[{idx}] 不是对象")
            continue
        category = item.get("category")
        polarity = item.get("polarity")
        subject = (item.get("subject") or "").strip()
        quote = item.get("quote") or ""
        if category not in _VALID_CATEGORIES:
            dropped.append(f"evidence[{idx}] category 非法：{category!r}")
            continue
        if polarity not in _VALID_POLARITIES:
            dropped.append(f"evidence[{idx}] polarity 非法：{polarity!r}")
            continue
        if not subject:
            dropped.append(f"evidence[{idx}] subject 为空")
            continue
        if _is_sensitive(subject):
            dropped.append(f"evidence[{idx}] subject 命中敏感属性，已剔除")
            continue
        if not _validate_quote(quote, raw_text):
            dropped.append(f"evidence[{idx}] quote 无法在原文定位")
            continue
        try:
            valid.append(Evidence(
                resume_id=resume_id,
                category=category,
                subject=subject,
                quote=quote,
                polarity=polarity,
            ))
        except (TypeError, ValueError) as exc:
            dropped.append(f"evidence[{idx}] 数据校验失败：{exc}")
    return valid, dropped


def _is_sensitive(subject: str) -> bool:
    folded = subject.strip().casefold()
    if folded in _SENSITIVE_EXACT_KEYWORDS:
        return True
    return any(kw.casefold() in folded for kw in _SENSITIVE_SUBJECT_KEYWORDS)


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _parse_extraction_json(text: str) -> dict[str, Any]:
    """解析 LLM 返回的 JSON；复用 core.parse_json，并做最小容错。"""
    from core import parse_json
    return parse_json(text)


def _has_positive_evidence(
    evidence: list[Evidence], category: str
) -> bool:
    return any(item.category == category and item.polarity == "positive" for item in evidence)


def _synthesize_grounded_evidence(
    evidence: list[Evidence],
    resume_id: str,
    raw_text: str,
    work_experience_years: float | None,
    education: str | None,
    warnings: list[str],
) -> list[Evidence]:
    """补齐被模型漏写的经验/学历证据，但 quote 必须能在原文逐字定位。

    仅补齐“结构化字段已抽到、原文存在对应字面值、模型漏写 Evidence”的情形，
    不推断原文没有的信息，也不改变 Matching Engine 的证据校验规则。
    """
    result = list(evidence)

    if work_experience_years is not None and not _has_positive_evidence(result, "experience"):
        number = f"{work_experience_years:g}"
        match = re.search(rf"(?<![\d.]){re.escape(number)}\s*年", raw_text)
        if match:
            result.append(Evidence(
                resume_id=resume_id,
                category="experience",
                subject="经验",
                quote=match.group(0),
                polarity="positive",
            ))
            warnings.append("模型漏写经验证据，已按原文中的年限补齐")

    if education and not _has_positive_evidence(result, "education") and education in raw_text:
        result.append(Evidence(
            resume_id=resume_id,
            category="education",
            subject="学历",
            quote=education,
            polarity="positive",
        ))
        warnings.append("模型漏写学历证据，已按原文中的学历字段补齐")

    return result


def build_profile_from_extraction(
    data: dict[str, Any],
    spec: ExtractionSpec,
) -> tuple[ResumeProfile, tuple[str, ...]]:
    """把 LLM 抽取的 dict 转换为 ResumeProfile，返回 (profile, warnings)。

    - skills / certifications 去重、去空、剔除敏感项。
    - explicitly_not_owned_* 仅用于后续显式否定信号，不写入 profile（由 orchestrator 转交）。
    - evidence 经清洗，quote 无法定位的直接丢弃。
    - 所有字段经过 screening_models 校验，校验失败抛 ValueError 由上层 fallback。
    """
    warnings: list[str] = []

    skills = _as_str_list(data.get("skills"))
    skills = _dedupe(_drop_sensitive(skills))

    certifications = _as_str_list(data.get("certifications"))
    certifications = _dedupe(_drop_sensitive(certifications))

    exp_raw = data.get("work_experience_years")
    if exp_raw is None or exp_raw == "":
        work_experience_years = None
    else:
        try:
            work_experience_years = float(exp_raw)
        except (TypeError, ValueError):
            warnings.append(f"工作年限无法解析：{exp_raw!r}，按未知处理")
            work_experience_years = None

    education = data.get("education")
    if education is not None and not isinstance(education, str):
        education = None
    education = (education or "").strip() or None
    if education is not None and _is_sensitive(education):
        warnings.append("学历字段命中敏感词，已置空")
        education = None

    evidence_items = data.get("evidence") or []
    evidence, dropped = _clean_evidence_items(evidence_items, spec.resume_id, spec.raw_text)
    warnings.extend(dropped)
    evidence = _synthesize_grounded_evidence(
        evidence,
        spec.resume_id,
        spec.raw_text,
        work_experience_years,
        education,
        warnings,
    )

    profile = ResumeProfile(
        resume_id=spec.resume_id,
        source_name=spec.source_name,
        candidate_name=None,
        skills=tuple(skills),
        work_experience_years=work_experience_years,
        education=education,
        certifications=tuple(certifications),
        work_history=(),
        evidence=tuple(evidence),
        extraction_warnings=tuple(warnings),
    )
    return profile, tuple(warnings)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _drop_sensitive(items: list[str]) -> list[str]:
    return [item for item in items if not _is_sensitive(item)]


def extract_explicit_not_owned(data: dict[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    """从抽取结果中取出明确否定信号集合（技能 / 证书）。"""
    skills = frozenset(_drop_sensitive(_dedupe(_as_str_list(data.get("explicitly_not_owned_skills")))))
    certs = frozenset(_drop_sensitive(_dedupe(_as_str_list(data.get("explicitly_not_owned_certifications")))))
    return skills, certs


def run_extraction(
    client: Any,
    spec: ExtractionSpec,
    *,
    max_retries: int = 2,
    agent_instruction: str | None = None,
) -> ExtractionOutcome:
    """执行一次结构化抽取，含重试与 fallback。

    client 需提供 complete(system, user) -> str（兼容 core.LLMClient 与 Mock）。
    agent_instruction 为 Registry 选中的 Agent 指令，非空时前置到 system prompt
    （不覆盖原有抽取规则与 JSON 输出要求），为空/None 时安全跳过。
    重试耗尽后返回 ok=False 的 ExtractionOutcome，绝不伪造成功结果。
    """
    system, user = build_extraction_prompt(spec)
    instr = (agent_instruction or "").strip()
    if instr:
        # 前置注入 Agent 指令，保留原有抽取规则、Evidence 约束与 JSON 要求。
        system = f"{instr}\n\n{system}"
    last_error: str | None = None
    for attempt in range(1, max_retries + 2):
        try:
            text = client.complete(system, user)
            data = _parse_extraction_json(text)
            profile, warnings = build_profile_from_extraction(data, spec)
            not_owned_skills, not_owned_certs = extract_explicit_not_owned(data)
            return ExtractionOutcome(
                resume_id=spec.resume_id, profile=profile, ok=True,
                warnings=warnings,
                explicit_not_owned_skills=not_owned_skills,
                explicit_not_owned_certifications=not_owned_certs,
                llm_request_attempts=attempt,
            )
        except Exception as exc:  # noqa: BLE001 - 记录任意异常供重试与 fallback
            last_error = f"{type(exc).__name__}: {exc}"
            continue
    return ExtractionOutcome(
        resume_id=spec.resume_id, profile=None, ok=False,
        error=last_error or "抽取失败",
        llm_request_attempts=max_retries + 1,
    )
