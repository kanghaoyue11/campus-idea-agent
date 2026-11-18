"""招聘筛选流程编排（Screening Orchestrator，招聘流水线第四阶段）。

把简历从「文本 -> 结构化抽取 -> Evidence 校验 -> 确定性评分 -> 排序」串成
一条可追踪的流水线。复用 core.LLMClient 与 core.RunLog，不重写 scheduler 的
并发/重试机制；评分完全交由冻结的 matching.score / matching.rank 完成，
LLM 不得覆盖任何评分结果。

可追踪阶段（stage）：
    1. resume_ingestion       简历输入登记
    2. resume_parsing         文本提取
    3. resume_extraction      LLM 结构化抽取
    4. evidence_validation    Evidence 原文定位校验（并入抽取清洗 + 评分前复验）
    5. deterministic_matching 调用 matching.score / matching.rank
    6. candidate_review       Reviewer Agent 复核建议
    7. result_aggregation     汇总报告

每阶段记录输入摘要、状态、起止时间、耗时、错误、重试次数、输出摘要；
日志不记录 API Key 与完整简历个人信息。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core import RunLog
from matching import rank, score
from screening_extraction import (
    ExtractionSpec,
    build_profile_from_extraction,
    extract_explicit_not_owned,
    run_extraction,
)
from screening_extractors import (
    ExtractedResume,
    ExtractionError,
    extract_all_resumes,
)
from screening_models import JobRequirements, MatchResult, ResumeProfile
from screening_scheduler import (
    run_extraction_batch,
    run_review_batch,
    persist_screening_trace,
)

# 流水线阶段名（顺序固定，用于日志与前端展示）。
STAGES = (
    "resume_ingestion",
    "resume_parsing",
    "resume_extraction",
    "evidence_validation",
    "deterministic_matching",
    "candidate_review",
    "result_aggregation",
)


@dataclass
class StageLog:
    """单个阶段的执行记录。"""
    stage: str
    status: str  # success / failed / skipped / warning
    input_summary: str
    output_summary: str
    started_ms: int
    duration_ms: int
    error: str | None = None
    retries: int = 0


@dataclass
class CandidateResult:
    """单个候选人的端到端结果。"""
    resume_id: str
    source_name: str
    match_result: MatchResult | None
    profile: ResumeProfile | None
    raw_text: str = ""
    explicit_not_owned_skills: frozenset[str] = frozenset()
    explicit_not_owned_certifications: frozenset[str] = frozenset()
    review: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.match_result is not None


@dataclass
class ScreeningReport:
    """一次批量筛选的完整报告。"""
    job_id: str
    title: str
    requirements: JobRequirements
    candidates: list[CandidateResult]
    ranking: list[CandidateResult]
    stage_logs: list[StageLog]
    errors: list[ExtractionError]
    mode: str
    created_at: str
    trace_id: str


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


class ScreeningOrchestrator:
    """编排招聘筛选流水线。

    依赖注入一个 client（需提供 complete(system, user) -> str），
    与一个可选 reviewer（复核建议生成器，见 screening_reviewer）。
    """

    def __init__(
        self,
        client: Any,
        reviewer: Any | None = None,
        *,
        max_retries: int = 2,
    ):
        self.client = client
        self.reviewer = reviewer
        self.max_retries = max_retries
        self._trace_id = ""

    # ---- 主入口 ----

    def run(
        self,
        requirements: JobRequirements,
        files: list[tuple[bytes, str]],
        *,
        trace_id: str | None = None,
    ) -> ScreeningReport:
        """执行一次完整批量筛选。

        files: [(bytes, filename), ...]，字节内容 + 原始文件名。
        返回 ScreeningReport；评分与排序由 matching 完成。
        """
        self._trace_id = trace_id or _make_trace_id()
        stage_logs: list[StageLog] = []
        started = _now_ms()

        # 1) resume_ingestion
        stage_logs.append(self._stage(
            "resume_ingestion", "success",
            f"收到 {len(files)} 份简历；岗位 {requirements.title}",
            f"开始登记，trace_id={self._trace_id}",
            started, _now_ms(),
        ))

        # 2) resume_parsing
        parse_start = _now_ms()
        resumes, errors = extract_all_resumes(files)
        stage_logs.append(self._stage(
            "resume_parsing", "success" if resumes else "warning",
            f"待解析 {len(files)} 份",
            f"成功 {len(resumes)} 份，失败 {len(errors)} 份",
            parse_start, _now_ms(),
            error=(f"全部解析失败" if not resumes and files else None),
        ))

        # 3) resume_extraction（批量并发） + 4) evidence_validation（逐份）
        extract_start = _now_ms()
        extract_results = run_extraction_batch(
            resumes,
            self.client,
            requirements,
            max_retries=1,
            scheduler_max_retries=0,
            batch_id=self._trace_id,
        )
        candidates = extract_results["candidates"]
        errors.extend(extract_results["extract_errors"])
        trace_records = list(extract_results["trace_records"])
        # evidence_validation：对成功抽取的简历记录该阶段（保持 7 阶段可追踪）。
        for cand in candidates:
            if cand.profile is not None:
                stage_logs.append(self._stage(
                    "evidence_validation", "success",
                    f"简历 {cand.resume_id[:12]} 校验证据",
                    f"有效 Evidence {len(cand.profile.evidence)} 条",
                    extract_start, _now_ms(),
                ))
        extract_ok = sum(1 for c in candidates if c.profile is not None)
        extract_failed = len(candidates) - extract_ok
        stage_logs.append(self._stage(
            "resume_extraction", "success" if extract_failed == 0 else "warning",
            f"批量抽取 {len(resumes)} 份简历",
            f"成功 {extract_ok} 份，失败 {extract_failed} 份",
            extract_start, _now_ms(),
            error=(f"{extract_failed} 份抽取失败" if extract_failed else None),
        ))

        # 5) deterministic_matching
        match_start = _now_ms()
        for cand in candidates:
            if cand.profile is not None:
                cand.match_result = self._match(requirements, cand)
        matched_count = sum(1 for c in candidates if c.match_result is not None)
        stage_logs.append(self._stage(
            "deterministic_matching", "success",
            f"对 {len(candidates)} 份简历评分",
            f"完成评分 {matched_count} 份（确定性引擎）",
            match_start, _now_ms(),
        ))

        # 6) candidate_review（批量并发）
        review_start = _now_ms()
        review_records: list = []
        if self.reviewer is not None:
            review_records = run_review_batch(
                candidates,
                self.reviewer,
                requirements,
                scheduler_max_retries=0,
                batch_id=self._trace_id,
            )
            trace_records.extend(review_records)
        reviewed = sum(1 for c in candidates if c.review is not None)
        stage_logs.append(self._stage(
            "candidate_review", "success",
            f"待复核 {matched_count} 份",
            f"生成复核建议 {reviewed} 份",
            review_start, _now_ms(),
        ))

        # 排序（评分 + 排序均由 matching.rank 完成）
        ranking = rank([c.match_result for c in candidates if c.match_result is not None])
        ranked_by_id = {r.resume_id: r for r in ranking}
        ordered_candidates = sorted(
            [c for c in candidates if c.match_result is not None],
            key=lambda c: ranking.index(ranked_by_id[c.resume_id]),
        )

        # 7) result_aggregation
        agg_start = _now_ms()
        report = ScreeningReport(
            job_id=requirements.job_id,
            title=requirements.title,
            requirements=requirements,
            candidates=candidates,
            ranking=ordered_candidates,
            stage_logs=stage_logs,
            errors=errors,
            mode=getattr(self.client, "mode", "未知模式"),
            created_at=_iso_now(),
            trace_id=self._trace_id,
        )
        stage_logs.append(self._stage(
            "result_aggregation", "success",
            f"候选 {len(candidates)} 份",
            f"排序完成 {len(ordered_candidates)} 份，报告生成",
            agg_start, _now_ms(),
        ))
        # 持久化本批次的完整 Trace（复用 core.RUNS_DIR，独立文件名）。
        persist_screening_trace(
            trace_records,
            task_id=self._trace_id,
            mode=getattr(self.client, "mode", "未知模式"),
            created_at=report.created_at,
        )
        return report

    # ---- 内部 ----

    def _process_resume_legacy(self, resume: ExtractedResume, stage_logs: list[StageLog]) -> CandidateResult:
        """（回滚备用）单份简历的抽取 + evidence 校验。

        新主流程已改用 screening_scheduler.run_extraction_batch 批量并发抽取，
        本方法不再被 run() 调用，仅保留作为回滚安全网。
        """
        spec = ExtractionSpec(
            resume_id=resume.resume_id,
            source_name=resume.source_name,
            raw_text=resume.raw_text,
        )

        ext_start = _now_ms()
        outcome = run_extraction(self.client, spec, max_retries=self.max_retries)
        stage_logs.append(self._stage(
            "resume_extraction",
            "success" if outcome.ok else "failed",
            f"简历 {resume.resume_id[:12]} 抽取结构化信息",
            f"{'成功' if outcome.ok else '失败：' + (outcome.error or '未知')}",
            ext_start, _now_ms(),
            error=(None if outcome.ok else outcome.error),
            retries=(0 if outcome.ok else self.max_retries + 1),
        ))

        if not outcome.ok or outcome.profile is None:
            return CandidateResult(
                resume_id=resume.resume_id,
                source_name=resume.source_name,
                match_result=None,
                profile=None,
                raw_text=resume.raw_text,
                warnings=outcome.warnings,
                error=outcome.error or "抽取失败",
            )

        profile = outcome.profile
        # evidence_validation：profile.evidence 已在抽取清洗阶段做过 quote 定位校验，
        # 此处再次显式记录该阶段（同一份已清洗证据），保持 7 阶段可追踪。
        ev_start = _now_ms()
        valid_count = len(profile.evidence)
        stage_logs.append(self._stage(
            "evidence_validation", "success",
            f"简历 {resume.resume_id[:12]} 校验证据",
            f"有效 Evidence {valid_count} 条",
            ev_start, _now_ms(),
        ))

        return CandidateResult(
            resume_id=resume.resume_id,
            source_name=resume.source_name,
            profile=profile,
            match_result=None,
            raw_text=resume.raw_text,
            explicit_not_owned_skills=outcome.explicit_not_owned_skills,
            explicit_not_owned_certifications=outcome.explicit_not_owned_certifications,
            warnings=outcome.warnings,
        )

    def _match(self, requirements: JobRequirements, cand: CandidateResult) -> MatchResult:
        """调用冻结的 matching.score，把抽取结果转交确定性评分。

        绝不在这里用 LLM 覆盖评分。raw_text 必须传入，否则 Evidence 无法定位校验。
        """
        return score(
            requirements,
            cand.profile,
            raw_text=cand.raw_text,
            explicitly_not_owned=cand.explicit_not_owned_skills,
            explicitly_not_owned_certifications=cand.explicit_not_owned_certifications,
        )

    @staticmethod
    def _stage(
        stage: str, status: str, input_summary: str, output_summary: str,
        started_ms: int, ended_ms: int, error: str | None = None, retries: int = 0,
    ) -> StageLog:
        return StageLog(
            stage=stage, status=status, input_summary=input_summary,
            output_summary=output_summary, started_ms=started_ms,
            duration_ms=ended_ms - started_ms, error=error, retries=retries,
        )


def _make_trace_id() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def _iso_now() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")
