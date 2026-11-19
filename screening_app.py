"""AI 招聘简历筛选网页端。

启动方式：streamlit run screening_app.py

功能：
    - 岗位要求配置（名称、描述、必备/加分技能、经验区间、学历、证书）。
    - PDF / TXT 简历批量上传。
    - 一键开始筛选，实时展示 7 个流水线阶段状态。
    - 候选人评分、排序、Evidence、人工复核意见、错误与日志。

评分完全由 Matching Engine v3.4 完成，LLM 仅做结构化抽取与复核建议，
不覆盖任何分数、推荐等级或排序。
"""

from __future__ import annotations

import json
import os

import streamlit as st

import matching as engine_matching
from core import LLMClient
from screening_extraction import run_extraction  # noqa: F401 - 保持模块可用
from screening_models import JobRequirements
from screening_orchestrator import ScreeningOrchestrator
from screening_reviewer import DeterministicReviewer, LLMReviewer


st.set_page_config(
    page_title="有据｜简历筛选",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 设计系统：暖纸色背景、深色文字与陶土橙强调色，形成简洁、易读的界面。
_TERRA = "#D97757"          # 陶土橙（唯一强调色）
_TERRA_DARK = "#C05F3E"
_PAPER = "#F5F4ED"          # 暖羊皮纸底
_INK = "#141413"            # 暖近黑
_INK_2 = "#6B6A63"          # 次文字（暖灰）
_BORDER = "rgba(20,20,19,0.10)"
_BG_CARD = "#FCFBF7"        # 卡片（比底稍亮、略暖）
_GREEN = "#5E7A5B"          # 成功（低饱和鼠尾草绿）
_ORANGE = "#B07A3E"         # 警示（暖琥珀）
_RED = "#A95C4F"            # 错误（低饱和陶红）

# 自绘“文档与核验”标记，使用内联 SVG。
_EVIDENCE_MARK_SVG = (
    "<svg width='24' height='24' viewBox='0 0 24 24' fill='none' "
    "xmlns='http://www.w3.org/2000/svg'>"
    "<path d='M6.5 3.5h7l4 4v12a1 1 0 0 1-1 1h-10a1 1 0 0 1-1-1v-15a1 1 0 0 1 1-1Z' "
    "stroke='#141413' stroke-width='1.5' stroke-linejoin='round'/>"
    "<path d='M13.5 3.8v4h4M8.5 12h6M8.5 15h3' stroke='#D97757' "
    "stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/>"
    "<path d='m13.4 17 1.3 1.3 2.8-3' stroke='#D97757' stroke-width='1.5' "
    "stroke-linecap='round' stroke-linejoin='round'/>"
    "</svg>"
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');

    /* ---------- 全局：纸张质感 + 噪点 + 暖色 ---------- */
    html, body, [data-testid="stAppViewContainer"] {{
        background: {_PAPER} !important;
        color: {_INK};
    }}
    [data-testid="stHeader"] {{
        background: rgba(245,244,237,0.72);
        backdrop-filter: saturate(140%) blur(20px);
    }}
    /* 全局纸张噪点（几乎不可见，模拟高级纸张触感） */
    [data-testid="stAppViewContainer"]::before {{
        content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.028'/%3E%3C/svg%3E");
    }}
    .block-container {{
        padding-top: 2.2rem; padding-bottom: 6rem; max-width: 1080px; position: relative; z-index: 1;
    }}
    h1, h2, h3, h4 {{
        color: {_INK}; letter-spacing: -0.02em; font-family: 'Instrument Serif', Georgia, serif;
        font-weight: 400;
    }}

    /* ---------- 顶部 Hero（衬线大标题 + 星号 Logo） ---------- */
    .hero {{ text-align: center; padding: 2.6rem 1.5rem 2.2rem; }}
    .hero .logo {{ display: inline-flex; align-items: center; gap: 0.55rem; margin-bottom: 1.4rem; }}
    .hero .logo svg {{ display: block; }}
    .hero .logo .word {{ font-family: 'Inter', sans-serif; font-weight: 500; letter-spacing: 0.02em; color: {_INK}; font-size: 0.95rem; }}
    .hero .eyebrow {{
        font-family: 'Inter', sans-serif; font-size: 0.76rem; font-weight: 500;
        letter-spacing: 0.22em; text-transform: uppercase; color: {_TERRA}; margin-bottom: 1rem;
    }}
    .hero h1 {{
        font-size: clamp(2.4rem, 6vw, 4rem); font-weight: 400; margin: 0 0 1.1rem;
        line-height: 1.02; letter-spacing: -0.03em; color: {_INK};
    }}
    .hero h1 em {{ font-style: italic; color: {_TERRA}; }}
    .hero .sub {{
        font-family: 'Inter', sans-serif; font-size: 1.02rem; font-weight: 300;
        color: {_INK_2}; max-width: 600px; margin: 0 auto; line-height: 1.7;
    }}

    /* ---------- 卡片（柔和阴影，无厚重阴影） ---------- */
    .card {{
        background: {_BG_CARD}; border: 1px solid {_BORDER};
        border-radius: 16px; padding: 1.5rem 1.7rem; margin: 1rem 0;
        box-shadow: 0 20px 50px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
    }}
    .card-title {{
        font-family: 'Instrument Serif', Georgia, serif; font-size: 1.3rem;
        color: {_INK}; margin-bottom: 0.3rem; letter-spacing: -0.01em;
    }}
    .card-sub {{ font-family: 'Inter', sans-serif; font-size: 0.88rem; font-weight: 300; color: {_INK_2}; margin-bottom: 1.1rem; }}

    /* ---------- 候选人大卡片 ---------- */
    .cand-head {{
        display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
        padding-bottom: 1rem; border-bottom: 1px solid {_BORDER}; margin-bottom: 1.1rem;
    }}
    .rank-badge {{
        flex: none; width: 46px; height: 46px; border-radius: 50%;
        background: {_TERRA}; color: #fff;
        font-family: 'Instrument Serif', serif; font-size: 1.35rem; line-height: 1;
        display: flex; align-items: center; justify-content: center;
    }}
    .cand-name {{ font-family: 'Instrument Serif', Georgia, serif; font-size: 1.4rem; color: {_INK}; }}
    .cand-meta {{ font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 300; color: {_INK_2}; }}

    .rec-pill {{
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.82rem;
        padding: 0.3rem 0.9rem; border-radius: 999px; white-space: nowrap;
    }}
    .rec-strong {{ color: #3F523C; background: rgba(94,122,91,0.14); }}
    .rec-potential {{ color: #8C5A32; background: rgba(217,119,87,0.14); }}
    .rec-limited {{ color: #8A463C; background: rgba(169,92,79,0.12); }}
    .rec-insufficient {{ color: #8C6A32; background: rgba(176,122,62,0.14); }}

    /* ---------- 统计条 ---------- */
    .stat-row {{ display: flex; gap: 2rem; flex-wrap: wrap; margin-top: 0.9rem; }}
    .stat {{ text-align: left; }}
    .stat .k {{ font-family: 'Inter', sans-serif; font-size: 0.74rem; font-weight: 400; color: {_INK_2}; letter-spacing: 0.05em; text-transform: uppercase; }}
    .stat .v {{ font-family: 'Instrument Serif', Georgia, serif; font-size: 1.9rem; color: {_INK}; }}
    .stat .v.strong {{ color: {_GREEN}; }}
    .stat .v.limited {{ color: {_RED}; }}
    .stat .v.insuf {{ color: {_ORANGE}; }}

    /* ---------- 维度得分表 ---------- */
    .dim-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    .dim-table th {{
        text-align: left; font-weight: 500; color: {_INK_2}; font-size: 0.76rem;
        padding: 0.6rem 0.75rem; border-bottom: 1px solid {_BORDER};
        font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .dim-table td {{ padding: 0.75rem; border-bottom: 1px solid {_BORDER}; vertical-align: middle; }}
    .dim-table tr:last-child td {{ border-bottom: none; }}
    .dim-name {{ font-family: 'Instrument Serif', Georgia, serif; font-size: 1.02rem; color: {_INK}; white-space: nowrap; }}
    .dim-weight {{ color: {_INK_2}; white-space: nowrap; font-family: 'Inter', sans-serif; }}
    .bar {{
        display: inline-block; width: 110px; height: 6px; border-radius: 999px;
        background: rgba(20,20,19,0.08); vertical-align: middle; margin-right: 0.6rem;
    }}
    .bar > span {{ display: block; height: 6px; border-radius: 999px; }}
    .dim-na {{ color: {_INK_2}; font-size: 0.84rem; font-family: 'Inter', sans-serif; }}
    .dim-reason {{ color: {_INK_2}; font-size: 0.86rem; line-height: 1.55; font-family: 'Inter', sans-serif; font-weight: 300; }}

    /* ---------- 匹配明细三栏 ---------- */
    .match-cols {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
    .match-col {{ flex: 1 1 200px; min-width: 180px; }}
    .match-col .col-head {{
        font-family: 'Inter', sans-serif; font-size: 0.78rem; font-weight: 500;
        letter-spacing: 0.05em; margin-bottom: 0.55rem;
    }}
    .col-ok .col-head {{ color: {_GREEN}; }}
    .col-fail .col-head {{ color: {_RED}; }}
    .col-unk .col-head {{ color: {_ORANGE}; }}
    .item {{
        padding: 0.55rem 0.8rem; margin: 0.45rem 0; border-radius: 10px;
        font-size: 0.85rem; line-height: 1.45; background: rgba(245,244,237,0.7);
        border: 1px solid {_BORDER}; font-family: 'Inter', sans-serif; font-weight: 300;
    }}
    .item-ok {{ border-left: 2px solid {_GREEN}; }}
    .item-fail {{ border-left: 2px solid {_RED}; }}
    .item-unk {{ border-left: 2px solid {_ORANGE}; }}
    .item-none {{ color: {_INK_2}; background: none; border: none; padding-left: 0; }}

    /* ---------- 一句话结论 / 提示条 ---------- */
    .gap-note {{
        padding: 0.85rem 1.1rem; border-radius: 12px; margin: 0.6rem 0;
        font-family: 'Inter', sans-serif; font-size: 0.9rem; font-weight: 300; line-height: 1.6; color: {_INK};
        background: rgba(217,119,87,0.07); border: 1px solid rgba(217,119,87,0.22);
    }}
    .warn-note {{
        background: rgba(176,122,62,0.08); border: 1px solid rgba(176,122,62,0.24);
    }}

    /* ---------- 证据 ---------- */
    .ev {{
        display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap;
        font-family: 'Inter', sans-serif; font-size: 0.86rem; font-weight: 300; padding: 0.55rem 0;
        border-bottom: 1px solid {_BORDER};
    }}
    .ev:last-child {{ border-bottom: none; }}
    .ev-cat {{
        display: inline-block; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.04em;
        padding: 0.1rem 0.6rem; border-radius: 999px; background: rgba(20,20,19,0.06); color: {_INK_2};
    }}
    .ev-pos {{ color: {_GREEN}; font-weight: 500; }}
    .ev-neg {{ color: {_RED}; font-weight: 500; }}
    .ev-subject {{ color: {_INK}; font-weight: 500; }}
    .ev-quote {{ color: {_INK_2}; }}

    /* ---------- 流水线 ---------- */
    .stage {{
        display: flex; align-items: center; gap: 0.8rem; padding: 0.7rem 0;
        border-bottom: 1px solid {_BORDER}; font-family: 'Inter', sans-serif;
    }}
    .stage:last-child {{ border-bottom: none; }}
    .stage-dot {{ flex: none; width: 8px; height: 8px; border-radius: 999px; }}
    .stage-dot.ok {{ background: {_GREEN}; }}
    .stage-dot.warn {{ background: {_ORANGE}; }}
    .stage-dot.fail {{ background: {_RED}; }}
    .stage-name {{ font-weight: 500; color: {_INK}; min-width: 8em; }}
    .stage-out {{ color: {_INK_2}; font-size: 0.86rem; font-weight: 300; }}

    /* ---------- 侧栏 ---------- */
    [data-testid="stSidebar"] {{ background: #FCFBF7; border-right: 1px solid {_BORDER}; }}
    [data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

    /* ---------- 按钮（微圆角、细边框、克制的陶土橙） ---------- */
    .stButton > button {{
        border-radius: 8px; font-weight: 500; border: 1px solid {_TERRA};
        background: {_TERRA}; color: #fff; padding: 0.55rem 1.1rem;
        font-family: 'Inter', sans-serif; letter-spacing: 0.01em;
        transition: all .3s cubic-bezier(0.16,1,0.3,1);
    }}
    .stButton > button:hover {{ background: {_TERRA_DARK}; border-color: {_TERRA_DARK}; transform: translateY(-1px); }}

    /* ---------- 输入框（细边框与轻圆角） ---------- */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input {{
        border-radius: 10px; border: 1px solid {_BORDER}; background: #fff;
    }}
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus, [data-testid="stNumberInput"] input:focus {{
        border-color: {_TERRA}; box-shadow: 0 0 0 3px rgba(217,119,87,0.12);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

REC_LABEL = {
    "strong_match": "强匹配",
    "potential_match": "潜在匹配",
    "limited_match": "有限匹配",
    "insufficient_information": "信息不足",
}
REC_CLASS = {
    "strong_match": "rec-strong",
    "potential_match": "rec-potential",
    "limited_match": "rec-limited",
    "insufficient_information": "rec-insufficient",
}


def _parse_skills(text: str) -> tuple[str, ...]:
    """把逗号/换行分隔的技能文本解析为去重元组。

    “无/没有/不限”等占位词按空值处理，避免被当成真实技能或证书。
    """
    import re

    none_values = {"无", "没有", "不限", "none", "n/a", "na", "null"}
    parts = [p.strip() for p in text.replace("，", ",").replace("\n", ",").split(",")]
    seen: list[str] = []
    for raw_part in parts:
        part = re.sub(r"\s*\+\s*", "+", raw_part).strip()
        if part and part.casefold() not in none_values and part not in seen:
            seen.append(part)
    return tuple(seen)


def _build_requirements(
    title: str,
    description: str,
    required: str,
    preferred: str,
    min_exp: float | None,
    max_exp: float | None,
    education: str,
    certifications: str,
) -> JobRequirements:
    return JobRequirements(
        job_id="job-" + str(abs(hash(title + description)) % 100000),
        title=title.strip() or "未命名岗位",
        required_skills=_parse_skills(required),
        preferred_skills=_parse_skills(preferred),
        min_experience_years=min_exp,
        max_experience_years=max_exp,
        education_requirement=(education.strip() or None),
        required_certifications=_parse_skills(certifications),
        responsibilities=(),
    )


def _run_simulation(orchestrator: ScreeningOrchestrator, req: JobRequirements, files):
    """模拟模式下，用本地确定性抽取逻辑替代真实 LLM，走完整流水线。

    抽取结果由简单的关键词规则生成，仅用于演示完整流程，不代表真实模型能力。
    """

    # 构造一个本地确定性 client：从 user 参数中取出简历原文，做关键词抽取。
    class _SimClient:
        mode = "模拟模式（不调用真实模型）"

        def complete(self, system, user):
            # user 形如 "简历原文：\n{raw_text}"
            raw = user.split("简历原文：", 1)[-1].strip()
            return _simulate_extraction_json(raw)

    from screening_orchestrator import ScreeningOrchestrator as SO
    sim_orch = SO(_SimClient(), reviewer=orchestrator.reviewer)
    return sim_orch.run(req, files)


def _parse_simulation_experience(raw: str) -> float | None:
    """从简历文本中识别工作年限，避免把毕业年份误当成工作年限。"""
    import re

    patterns = (
        r"(?:工作年限|工作经验|从业年限)\s*[：:为]?\s*(\d+(?:\.\d+)?)\s*年",
        r"(\d+(?:\.\d+)?)\s*年(?:以上)?(?:相关|工作|开发|从业|项目)?经验",
        r"(\d+(?:\.\d+)?)\s*年[^\n。；]{0,20}?经验",
        r"(?:工作|开发|从业)\s*(\d+(?:\.\d+)?)\s*年",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, raw):
            value = float(match.group(1))
            if 0 <= value <= 60:
                return value
    return None


def _contains_simulation_keyword(raw: str, keyword: str) -> bool:
    """按英文词边界匹配技能，避免把 JavaScript 识别成 Java。"""
    import re

    pattern = rf"(?<![A-Za-z0-9+#]){re.escape(keyword)}(?![A-Za-z0-9+#])"
    return re.search(pattern, raw, re.IGNORECASE) is not None


def _simulate_extraction_json(raw: str) -> str:
    """极简本地抽取：识别少量技能/学历关键词，生成 JSON。仅用于模拟演示。"""
    skills = []
    for kw in (
        "Python", "FastAPI", "Docker", "Kubernetes", "Java", "JavaScript",
        "TypeScript", "C++", "C#", "Go", "React", "Vue", "Node.js",
        "SQL", "MySQL", "PostgreSQL", "Redis",
    ):
        if _contains_simulation_keyword(raw, kw):
            skills.append(kw)
    not_owned = []
    for kw in ("Kubernetes", "AWS", "PMP"):
        if f"不会 {kw}" in raw or f"无 {kw}" in raw or f"不会{kw}" in raw:
            not_owned.append(kw)
    education = None
    for edu in ("博士", "硕士", "本科", "大专", "专科", "高中"):
        if edu in raw:
            education = edu
            break
    # 只识别明确的经验表达，跳过“2026年”这类日期/毕业年份。
    exp = _parse_simulation_experience(raw)
    certifications = []
    for cert in ("PMP", "AWS"):
        if cert in raw and f"无 {cert}" not in raw:
            certifications.append(cert)
    evidence = []
    for s in skills:
        evidence.append({"category": "skill", "subject": s, "quote": s, "polarity": "positive"})
    for s in not_owned:
        evidence.append({"category": "skill", "subject": s, "quote": f"不会 {s}", "polarity": "negative"})
    if exp is not None:
        evidence.append({"category": "experience", "subject": "经验", "quote": f"{int(exp)}年", "polarity": "positive"})
    if education:
        evidence.append({"category": "education", "subject": education, "quote": education, "polarity": "positive"})
    for c in certifications:
        evidence.append({"category": "certification", "subject": c, "quote": c, "polarity": "positive"})
    payload = {
        "skills": skills,
        "work_experience_years": exp,
        "education": education,
        "certifications": certifications,
        "explicitly_not_owned_skills": not_owned,
        "explicitly_not_owned_certifications": [],
        "evidence": evidence,
    }
    return json.dumps(payload, ensure_ascii=False)


def _report_to_json(report) -> str:
    candidates = []
    for c in report.candidates:
        mr = c.match_result
        candidates.append({
            "resume_id": c.resume_id,
            "source_name": c.source_name,
            "overall_score": mr.overall_score if mr else None,
            "recommendation": mr.recommendation if mr else None,
            "information_completeness": mr.information_completeness if mr else None,
            "unknown_items": list(mr.unknown_items) if mr else [],
            "not_matched_items": list(mr.not_matched_items) if mr else [],
            "dimension_scores": mr.dimension_scores if mr else None,
            "evidence": [
                {"category": e.category, "subject": e.subject, "quote": e.quote, "polarity": e.polarity}
                for e in (mr.evidence if mr else [])
            ],
            "review": c.review,
            "warnings": list(c.warnings),
            "error": c.error,
        })
    payload = {
        "job_id": report.job_id,
        "title": report.title,
        "requirements": {
            "required_skills": list(report.requirements.required_skills),
            "preferred_skills": list(report.requirements.preferred_skills),
            "min_experience_years": report.requirements.min_experience_years,
            "max_experience_years": report.requirements.max_experience_years,
            "education_requirement": report.requirements.education_requirement,
            "required_certifications": list(report.requirements.required_certifications),
        },
        "candidates": candidates,
        "stage_logs": [
            {"stage": s.stage, "status": s.status, "duration_ms": s.duration_ms, "output": s.output_summary}
            for s in report.stage_logs
        ],
        "mode": report.mode,
        "trace_id": report.trace_id,
        "created_at": report.created_at,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ==== 展示层：把引擎的英文枚举/字段翻译成中文，不改动任何评分结果 ====

DIM_LABEL = {
    "required": "必备技能",
    "preferred": "加分技能",
    "experience": "工作经验",
    "education": "学历",
    "certification": "证书",
}
DIM_ORDER = ("required", "preferred", "experience", "education", "certification")
# 权重直接取自评分引擎，避免页面说明与实际规则不一致。
DIM_WEIGHT = getattr(
    engine_matching,
    "_WEIGHTS",
    {"required": 0.50, "preferred": 0.10, "experience": 0.20, "education": 0.10, "certification": 0.10},
)

STAGE_ORDER = (
    "resume_ingestion",
    "resume_parsing",
    "resume_extraction",
    "evidence_validation",
    "deterministic_matching",
    "candidate_review",
    "result_aggregation",
)
STAGE_LABEL = {
    "resume_ingestion": "简历登记",
    "resume_parsing": "文本提取",
    "resume_extraction": "结构化抽取",
    "evidence_validation": "证据校验",
    "deterministic_matching": "确定性评分",
    "candidate_review": "复核建议",
    "result_aggregation": "结果汇总",
}
STATUS_LABEL = {"success": "完成", "warning": "有告警", "failed": "失败", "error": "失败"}

CATEGORY_LABEL = {
    "skill": "技能",
    "experience": "经验",
    "education": "学历",
    "certification": "证书",
}
POLARITY_LABEL = {"positive": "支持", "negative": "反证"}

STATUS_TEXT = {"matched": "已具备", "not_matched": "明确不具备", "unknown": "简历未提及"}
EXP_TEXT = {"met": "符合", "not_met": "不满足", "unknown": "未判定"}
EDU_TEXT = {"met": "符合", "not_met": "不满足", "unknown": "未判定"}


def _plain_error(text: str) -> str:
    """去掉异常类名等程序痕迹，只留中文原因。"""
    msg = str(text)
    for token in ("ExtractionFailedError: ", "ExtractionError: ", "ValueError: ", "TypeError: "):
        msg = msg.replace(token, "")
    msg = msg.replace("work_experience_years", "工作年限")
    return msg.strip()


def _humanize(text: str) -> str:
    """把复核意见、阶段摘要里残留的英文字段名换成中文说法。"""
    import re

    msg = str(text)
    replacements = (
        ("unknown_items", "未判定项"),
        ("not_matched_items", "不满足项"),
        ("dimension_scores", "各维度得分"),
        ("information_completeness", "信息完整度"),
        ("overall_score", "综合分"),
        ("evidence_count", "证据条数"),
        ("Evidence", "证据"),
        ("trace_id", "运行编号"),
        ("resume-", "简历编号 "),
    )
    for old, new in replacements:
        msg = msg.replace(old, new)
    # “加分技能:Docker” → “加分技能「Docker」”
    msg = re.sub(r"(必备技能|加分技能|证书)[:：]\s*([^\s；;，,]+)", r"\1「\2」", msg)
    # 抽取告警里的技术描述改成普通话
    msg = re.sub(r"evidence\[\d+\]\s*subject 命中敏感属性，已剔除", "含敏感信息的证据已自动剔除", msg)
    msg = re.sub(r"有效\s*证据", "有效证据", msg)
    msg = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", msg)
    return msg.strip()


def _short_name(text: str) -> str:
    """把“工作经验：简历 1 年…”压缩成“工作经验”，用于一句话结论。"""
    return str(text).split("：", 1)[0]


def _review_block(cand, mr, ok: list[str], fail: list[str], unknown: list[str]) -> tuple[str, list[str]]:
    """生成一句中文复核结论；技术性提示另外列出，避免整段术语堆叠。"""
    review = cand.review or {}
    summary = str(review.get("summary", "") or "")
    flags = [_humanize(f) for f in review.get("flags", [])]
    is_template = summary.startswith("建议人工复核：") or summary.startswith("未发现需额外关注")

    if not is_template:
        # LLM 复核给出的自然语言意见，原样保留。
        return summary, flags

    fail_names = "、".join(_short_name(x) for x in fail)
    unknown_names = "、".join(_short_name(x) for x in unknown)
    if mr.recommendation == "strong_match":
        conclusion = "岗位要求全部满足，建议进入下一轮筛选。"
    elif mr.recommendation == "potential_match":
        conclusion = "整体符合岗位要求，建议面试进一步确认。"
    elif mr.recommendation == "limited_match":
        conclusion = f"存在明确不满足项（{fail_names}），建议谨慎推进或转其他岗位。"
    else:
        conclusion = "关键信息缺失，暂时难以判断，建议补充材料后再决定。"
    if unknown and mr.recommendation != "insufficient_information":
        conclusion += f"另有 {len(unknown)} 项信息缺失（{unknown_names}），建议面试时核实。"

    notes = [
        f for f in flags
        if f.startswith("抽取告警") or f.startswith("抽取异常") or "没有任何有效" in f
    ]
    return conclusion, notes


def _fmt_num(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _years_range(lo, hi) -> str:
    if lo is not None and hi is not None:
        return f"{_fmt_num(lo)}–{_fmt_num(hi)} 年"
    if lo is not None:
        return f"至少 {_fmt_num(lo)} 年"
    if hi is not None:
        return f"不超过 {_fmt_num(hi)} 年"
    return "不限"


def _score_bar(score) -> str:
    """得分条：未判定维度明确标灰，避免被误解成 0 分。"""
    if score is None:
        return "<span class='dim-na'>未判定 · 不计入综合分</span>"
    value = max(0.0, min(100.0, float(score)))
    # 用陶土橙做主进度色，低分才转为警示色，保持整体克制。
    color = _GREEN if value >= 80 else _TERRA if value >= 60 else _ORANGE if value >= 40 else _RED
    return (
        f"<div class='bar'><span style='width:{value:.0f}%;background:{color}'></span></div>"
        f"<b>{_fmt_num(score)}</b>"
    )


def _education_verdict(mr) -> str:
    """MatchResult 没有单独的学历状态字段，用未判定/不满足清单反推。"""
    if "学历" in set(mr.not_matched_items or ()):
        return "not_met"
    if "学历" in set(mr.unknown_items or ()):
        return "unknown"
    return "met"


def _dim_reason(dim: str, mr, cand, req) -> str:
    """给每个维度一句中文依据，说明分数是怎么来的。"""
    profile = cand.profile
    if dim in ("required", "preferred"):
        statuses = (mr.skill_statuses or {}).get(dim, {}) or {}
        if not statuses:
            return "岗位未设置该项"
        ok = sum(1 for v in statuses.values() if v == "matched")
        bad = sum(1 for v in statuses.values() if v == "not_matched")
        unk = sum(1 for v in statuses.values() if v == "unknown")
        parts = [f"共 {len(statuses)} 项，已具备 {ok} 项"]
        if bad:
            parts.append(f"明确不具备 {bad} 项")
        if unk:
            parts.append(f"简历未提及 {unk} 项")
        return "，".join(parts)

    if dim == "experience":
        if req.min_experience_years is None and req.max_experience_years is None:
            return "岗位未要求工作经验"
        actual = profile.work_experience_years if profile else None
        actual_txt = f"{_fmt_num(actual)} 年" if actual is not None else "简历未写明"
        judge = EXP_TEXT.get(mr.experience_status, "未判定")
        return f"简历 {actual_txt}，要求 {_years_range(req.min_experience_years, req.max_experience_years)}（{judge}）"

    if dim == "education":
        if not req.education_requirement:
            return "岗位未要求学历"
        actual = (profile.education if profile and profile.education else "简历未写明")
        judge = EDU_TEXT.get(_education_verdict(mr), "未判定")
        return f"简历 {actual}，要求 {req.education_requirement}（{judge}）"

    certs = getattr(profile, "certifications", ()) if profile else ()
    if not req.required_certifications:
        return "岗位未要求证书"
    require_txt = "、".join(req.required_certifications)
    if certs:
        return f"简历有 {'、'.join(certs)}，要求 {require_txt}"
    return f"简历未写明证书，要求 {require_txt}"


def _match_groups(mr, cand, req) -> tuple[list[str], list[str], list[str]]:
    """把判定结果拆成「明确满足 / 明确不满足 / 信息不足」三组中文句子。"""
    profile = cand.profile
    ok: list[str] = []
    fail: list[str] = []
    unknown: list[str] = []

    for dim, label in (("required", "必备技能"), ("preferred", "加分技能")):
        for skill, status in ((mr.skill_statuses or {}).get(dim, {}) or {}).items():
            text = f"{label}「{skill}」"
            if status == "matched":
                ok.append(text)
            elif status == "not_matched":
                fail.append(f"{text}：简历明确写明不具备")
            else:
                unknown.append(f"{text}：简历未提及")

    if req.min_experience_years is not None or req.max_experience_years is not None:
        actual = profile.work_experience_years if profile else None
        actual_txt = f"{_fmt_num(actual)} 年" if actual is not None else "简历未写明"
        text = f"工作经验：简历 {actual_txt}，岗位要求 {_years_range(req.min_experience_years, req.max_experience_years)}"
        if mr.experience_status == "met":
            ok.append(text)
        elif mr.experience_status == "not_met":
            fail.append(text)
        else:
            unknown.append(text)

    if req.education_requirement:
        actual = profile.education if profile and profile.education else "简历未写明"
        text = f"学历：简历 {actual}，岗位要求 {req.education_requirement}"
        verdict = _education_verdict(mr)
        if verdict == "met":
            ok.append(text)
        elif verdict == "not_met":
            fail.append(text)
        else:
            unknown.append(text)

    unknown_items = set(mr.unknown_items or ())
    not_matched_items = set(mr.not_matched_items or ())
    for cert in req.required_certifications:
        text = f"证书「{cert}」"
        if f"证书:{cert}" in not_matched_items:
            fail.append(f"{text}：简历明确写明不具备")
        elif f"证书:{cert}" in unknown_items:
            unknown.append(f"{text}：简历未提及")
        else:
            ok.append(text)

    return ok, fail, unknown


def _render_dimensions(mr, cand, req) -> None:
    rows = []
    for dim in DIM_ORDER:
        score = (mr.dimension_scores or {}).get(dim)
        weight = DIM_WEIGHT.get(dim, 0) * 100
        rows.append(
            "<tr>"
            f"<td class='dim-name'>{DIM_LABEL[dim]}</td>"
            f"<td class='dim-weight'>{weight:g}%</td>"
            f"<td>{_score_bar(score)}</td>"
            f"<td>{_dim_reason(dim, mr, cand, req)}</td>"
            "</tr>"
        )
    st.markdown(
        "<table class='dim-table'>"
        "<thead><tr><th>评分维度</th><th>权重</th><th>得分</th><th>判定依据</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption(
        "未判定表示简历里没有足够证据（不是 0 分），不参与综合分计算；"
        "综合分只在已判定的维度之间按权重重新分配。"
    )


def _render_match_groups(ok: list[str], fail: list[str], unknown: list[str]) -> None:
    def _col(head, items, kind, empty="无"):
        head_html = f"<div class='col-head'>{head}</div>"
        if items:
            body = "".join(f"<div class='item item-{kind}'>{x}</div>" for x in items)
        else:
            body = f"<div class='item item-none'>{empty}</div>"
        return f"<div class='match-col col-{kind}'>{head_html}{body}</div>"

    st.markdown(
        "<div class='match-cols'>"
        + _col("已满足", ok, "ok")
        + _col("不满足", fail, "fail")
        + _col("待核实", unknown, "unk", empty="无信息缺口")
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_evidence(evidence) -> None:
    lines = []
    for ev in evidence:
        cat = CATEGORY_LABEL.get(ev.category, ev.category)
        polarity = POLARITY_LABEL.get(ev.polarity, ev.polarity)
        cls = "ev-pos" if ev.polarity == "positive" else "ev-neg"
        lines.append(
            f"<div class='ev'><span class='ev-cat'>{cat}</span>"
            f"<span class='{cls}'>{polarity}</span>"
            f"<span class='ev-subject'>{ev.subject}</span>"
            f"<span class='ev-quote'>「{ev.quote}」</span></div>"
        )
    st.markdown("".join(lines), unsafe_allow_html=True)


def _aggregate_stages(stage_logs) -> list[dict]:
    """同名阶段合并展示（证据校验会对每份简历各记一条）。"""
    merged: dict[str, dict] = {}
    for log in stage_logs:
        entry = merged.get(log.stage)
        if entry is None:
            merged[log.stage] = {
                "stage": log.stage,
                "status": log.status,
                "duration_ms": log.duration_ms,
                "count": 1,
                "output": log.output_summary,
            }
        else:
            entry["duration_ms"] += log.duration_ms
            entry["count"] += 1
            entry["output"] = log.output_summary
            if entry["status"] == "success" and log.status != "success":
                entry["status"] = log.status
    order = {name: i for i, name in enumerate(STAGE_ORDER)}
    return sorted(merged.values(), key=lambda item: order.get(item["stage"], len(order)))


def _comp_class(completeness) -> str:
    """信息完整度颜色（用于统计数字）：>=80 绿、>=60 橙、否则红；未判定返回空。"""
    if completeness is None:
        return ""
    if completeness >= 80:
        return "strong"
    if completeness >= 60:
        return "insuf"
    return "limited"


def _gap_summary(mr, ok, fail, unknown) -> str:
    """一句话差距总结：先说满足什么，再说缺什么/不满足什么。"""
    def _names(items):
        names = []
        for x in items:
            n = _short_name(x)
            # 技能项形如「必备技能「Java」」，只保留前缀“必备技能”
            if "「" in n:
                n = n.split("「", 1)[0]
            names.append(n)
        # 去重保序，避免“必备技能”重复出现
        seen = []
        for n in names:
            if n not in seen:
                seen.append(n)
        return "、".join(seen[:5])

    parts = []
    if ok:
        parts.append("已满足" + _names(ok))
    if fail:
        parts.append("不满足" + _names(fail))
    if unknown:
        parts.append("未提及" + _names(unknown))
    if not parts:
        return "岗位要求已全部覆盖"
    return "；".join(parts)


st.markdown(
    """
    <div class="hero">
      <div class="logo">{_EVIDENCE_MARK_SVG}<span class="word">有据 · 简历筛选</span></div>
      <div class="eyebrow">证据支持判断，人工保留决定权</div>
      <h1>让每一次筛选<br>都有<em>据</em>可依</h1>
      <div class="sub">
        上传简历，系统自动完成结构化抽取、证据校验与确定性评分，
        并用清晰的中文结果告诉你「为什么这样判断」。评分引擎不依赖大模型主观打分。
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 运行设置")
    mode = st.radio(
        "运行方式",
        ["模拟模式（不调用真实模型）", "真实模型 API"],
        horizontal=False,
        key="screening_mode",
    )
    is_simulation = mode.startswith("模拟")
    base_url = st.text_input(
        "兼容接口地址",
        value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        key="screening_base_url",
    )
    model = st.text_input(
        "模型名称",
        value=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        key="screening_model",
    )
    use_llm_review = st.checkbox("使用 LLM 复核建议", value=False, disabled=is_simulation)
    has_key = bool(os.getenv("OPENAI_API_KEY", ""))
    if is_simulation:
        st.caption("模拟模式使用本地固定抽取逻辑，不调用真实模型。")
    elif has_key:
        st.caption("已从 OPENAI_API_KEY 读取密钥，页面不显示。")
    else:
        st.warning("未检测到 OPENAI_API_KEY。")

# ---- 岗位要求配置（用 expander 包裹，可展开/收起，避免裸 div 包不住控件） ----
st.markdown("## 岗位要求")
with st.expander("设定筛选标准", expanded=True):
    st.caption("填写岗位的核心要求，系统将据此对每份简历逐项判定。")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("岗位名称", placeholder="例如：后端工程师", key="job_title")
        description = st.text_area("岗位描述", placeholder="简要描述岗位职责与要求", key="job_desc", height=90)
        required = st.text_area(
            "必备技能（逗号分隔）", placeholder="Python, FastAPI", key="job_required", height=80
        )
    with col2:
        preferred = st.text_area(
            "加分技能（逗号分隔）", placeholder="Docker, Kubernetes", key="job_preferred", height=80
        )
        exp_col_a, exp_col_b = st.columns(2)
        with exp_col_a:
            min_exp = st.number_input("最低经验（年）", min_value=0, max_value=60, value=3, key="job_min_exp")
        with exp_col_b:
            max_exp = st.number_input("最高经验（年）", min_value=0, max_value=60, value=8, key="job_max_exp")
        education = st.text_input("学历要求", placeholder="本科 / 硕士（留空表示不限）", key="job_edu")
        certifications = st.text_input("证书要求（逗号分隔）", placeholder="PMP", key="job_certs")

# ---- 简历上传 ----
st.markdown("## 简历上传")
with st.container(border=True):
    uploaded = st.file_uploader(
        "上传 PDF / TXT 简历（可多选）",
        type=["pdf", "txt", "text", "md"],
        accept_multiple_files=True,
        key="resume_upload",
    )
    run_clicked = st.button("开始筛选", type="primary", use_container_width=True)

if run_clicked:
    if not title.strip():
        st.error("请先填写岗位名称。")
    elif not uploaded:
        st.error("请至少上传一份简历。")
    else:
        files = [(f.getvalue(), f.name) for f in uploaded]
        req = _build_requirements(
            title, description, required, preferred,
            float(min_exp), float(max_exp), education, certifications,
        )
        client = LLMClient(mode, base_url, model, os.getenv("OPENAI_API_KEY", ""))
        reviewer = (
            LLMReviewer(client) if (use_llm_review and not is_simulation)
            else DeterministicReviewer()
        )
        orchestrator = ScreeningOrchestrator(client, reviewer=reviewer)

        # 模拟模式下，抽取由本地确定性逻辑兜底（不调用真实模型）
        with st.spinner("正在执行招聘筛选流水线…"):
            try:
                if is_simulation:
                    report = _run_simulation(orchestrator, req, files)
                else:
                    report = orchestrator.run(req, files)
            except Exception as exc:  # noqa: BLE001 - 前端友好提示
                st.error(f"筛选失败：{exc}")
                report = None

        if report is not None:
            st.session_state["screening_report"] = report

report = st.session_state.get("screening_report")

if report is not None:
    # ---- 流水线执行情况 ----
    st.markdown("## 处理过程")
    with st.container(border=True):
        st.markdown("<div class='card-title'>流水线执行情况</div>", unsafe_allow_html=True)
        stage_rows = _aggregate_stages(report.stage_logs)
        stage_lines = []
        for row in stage_rows:
            status = row["status"]
            dot_cls = "ok" if status == "success" else ("warn" if status == "warning" else "fail")
            status_txt = STATUS_LABEL.get(status, status)
            stage_lines.append(
                f"<div class='stage'><span class='stage-dot {dot_cls}'></span>"
                f"<span class='stage-name'>{STAGE_LABEL.get(row['stage'], row['stage'])}</span>"
                f"<span class='stage-out'>{status_txt} · {_humanize(row['output'])}</span></div>"
            )
        st.markdown("".join(stage_lines), unsafe_allow_html=True)

    # ---- 候选人排序结果 ----
    st.markdown("## 排序结果")
    if report.ranking:
        for idx, cand in enumerate(report.ranking, 1):
            mr = cand.match_result
            rec = mr.recommendation
            score_txt = f"{mr.overall_score}" if mr.overall_score is not None else "—"
            comp = mr.information_completeness
            comp_txt = f"{comp}%" if comp is not None else "—"
            comp_cls = _comp_class(comp)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            # 卡片头部：名次徽章 + 姓名 + 推荐结论
            st.markdown(
                "<div class='cand-head'>"
                f"<div class='rank-badge'>{idx}</div>"
                f"<div style='flex:1;min-width:160px;'>"
                f"<div class='cand-name'>{cand.source_name}</div>"
                f"<div class='cand-meta'>简历编号 {cand.resume_id.replace('resume-', '')}</div>"
                f"</div>"
                f"<span class='rec-pill {REC_CLASS[rec]}'>{REC_LABEL[rec]}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            # 统计条：综合分 + 信息完整度
            st.markdown(
                "<div class='stat-row'>"
                f"<div class='stat'><div class='k'>综合分</div><div class='v'>{score_txt}</div></div>"
                f"<div class='stat'><div class='k'>信息完整度</div><div class='v {comp_cls}'>{comp_txt}</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if comp is not None and comp < 100:
                st.markdown(
                    f"<div class='gap-note warn-note'>⚠️ 该分数仅基于 <b>{comp_txt}</b> 的简历信息得出，"
                    f"剩余 {_fmt_num(100 - comp)}% 岗位要求尚未得到证据，分数仅供参考。</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("各维度评分与依据", expanded=True):
                _render_dimensions(mr, cand, report.requirements)

            ok_items, fail_items, unknown_items = _match_groups(mr, cand, report.requirements)
            st.markdown("**一句话结论**")
            st.markdown(
                f"<div class='gap-note'>{_gap_summary(mr, ok_items, fail_items, unknown_items)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**匹配明细**")
            _render_match_groups(ok_items, fail_items, unknown_items)
            if unknown_items:
                st.caption("“待核实”只表示简历里没有写明，不等于不符合，建议面试或补充材料后确认。")

            if mr.evidence:
                with st.expander(f"评分依据（{len(mr.evidence)} 条原文证据）", expanded=False):
                    _render_evidence(mr.evidence)

            if cand.review:
                with st.expander("人工复核建议", expanded=True):
                    conclusion, notes = _review_block(cand, mr, ok_items, fail_items, unknown_items)
                    st.markdown(f"**结论**：{conclusion}")
                    for note in notes:
                        st.caption(note)
    else:
        st.warning("没有成功评分的候选人，请查看下方的解析失败原因。")

    # ---- 错误与解析失败 ----
    if report.errors:
        st.markdown("## 解析失败")
        with st.container(border=True):
            for err in report.errors:
                st.markdown(f"- **{err.source_name}**：{_plain_error(err.error)}")

    # ---- 下载报告 ----
    st.markdown("## 报告导出")
    report_json = _report_to_json(report)
    st.download_button(
        "下载完整报告（JSON，供程序读取）",
        report_json,
        file_name="screening_report.json",
        mime="application/json",
    )

    st.caption(
        f"运行方式：{report.mode} ｜ 运行编号：{report.trace_id} ｜ 生成时间：{report.created_at}"
    )
