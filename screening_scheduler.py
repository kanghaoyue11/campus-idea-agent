"""招聘筛选双阶段批量多 Agent 调度（第六阶段正式实施）。

把 Screening Pipeline 的「结构化抽取」与「复核」两个阶段，从逐份串行升级为
批次级并发调度，复用 core/scheduler 已有的 ``run_roles_concurrently`` 与
``select_agent``，不复制、不改写 Matching Engine 的评分与排序。

关键约定（与第五、六阶段冻结方案一致）：
    - agent_role（语义角色）：``extractor`` / ``screening_reviewer``。
    - stage（业务阶段）：``resume_extraction`` / ``candidate_review``。
    - task_key = f"{stage}:{resume_id}"，唯一标识一份简历在某个阶段的任务。
    - RoleResult.role 保存 task_key，不是 agent_role。
    - 每个阶段、每个批次只调用一次 select_agent（选型粒度 = 批次级）。
    - AgentAssignment.instruction 必须注入真实 LLM system prompt；
      instruction 为空时安全跳过，绝不把 None 拼入 prompt。
    - AgentAssignment.agent_id（Registry 真实 ID）写入 Trace。

预算（冻结）：
    - Extractor LLM max_retries = 1（每份简历最多 2 次 LLM 请求）。
    - Scheduler max_retries = 0（每个任务只执行 1 次）。
    - Reviewer LLM 请求 = 1 次（无重试）。
    - 完整成功每份简历最多 3 次 LLM 请求。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scheduler import run_roles_concurrently, select_agent

# 复用 core 的 Trace 存储目录约定（不另起一套目录）。
from core import RUNS_DIR

# 与 screening_orchestrator 的 STAGES 对齐的阶段名。
STAGE_EXTRACTION = "resume_extraction"
STAGE_REVIEW = "candidate_review"

# 并发上限：每个分片最多并发 4 个任务。
DEFAULT_SHARD_SIZE = 4


class ExtractionFailedError(RuntimeError):
    """抽取业务失败（ExtractionOutcome.ok == False）时抛出的受控异常。

    继承 RuntimeError，从而被 scheduler._run_one 的 ``except Exception``
    捕获，使 Scheduler 产出 ``RoleResult(ok=False)``，不会把业务失败伪装成
    调度成功。携带原始 outcome，供上层读取错误原因与请求次数。
    """

    def __init__(self, outcome: "ExtractionOutcome"):
        self.outcome = outcome
        super().__init__(outcome.error or "结构化抽取失败")


def _job_context(requirements) -> str:
    """由岗位要求构造 select_agent 的 context 文本。"""
    return f"{getattr(requirements, 'title', '')} {getattr(requirements, 'job_id', '')}"


def _safe_instruction(assignment) -> str:
    """安全取出 instruction，None/空字符串一律返回空串，绝不拼入 None。"""
    instr = getattr(assignment, "instruction", "") or ""
    return instr.strip()


def _shard_keys(keys: list[str], shard_size: int) -> list[list[str]]:
    """把 task_key 列表切成每个不超过 shard_size 的分片。"""
    if shard_size <= 0:
        raise ValueError("shard_size 必须大于 0。")
    return [keys[i:i + shard_size] for i in range(0, len(keys), shard_size)]


def _extraction_task(
    resume,
    client: Any,
    assignment,
    max_retries: int,
    attempts_by_key: dict[str, int] | None = None,
) -> tuple["ExtractionOutcome", str]:
    """单份简历的抽取任务闭包：注入 instruction、检查 ok、受控上抛。

    返回 (outcome, agent_id)。agent_id 取自 Registry 真实选出的实例 ID，
    供上层写入 Trace。抽取业务失败时抛 ExtractionFailedError，由 Scheduler 捕获。

    attempts_by_key（可选）为类型安全的数据通道：无论成功还是失败，都把本次
    抽取的真实 LLM 请求次数按 task_key 记录，供上层在失败场景（异常对象被
    Scheduler 转成字符串后）仍能精确取回请求次数，而不是记 0。
    """
    from screening_extraction import ExtractionSpec, build_extraction_prompt, run_extraction

    spec = ExtractionSpec(
        resume_id=resume.resume_id,
        source_name=resume.source_name,
        raw_text=resume.raw_text,
    )

    outcome = run_extraction(
        client,
        spec,
        max_retries=max_retries,
        agent_instruction=_safe_instruction(assignment),
    )

    # 无论成败，先把真实请求次数记录到数据通道（类型安全，非字符串解析）。
    if attempts_by_key is not None:
        attempts_by_key[resume.resume_id] = outcome.llm_request_attempts

    if not outcome.ok:
        raise ExtractionFailedError(outcome)

    return outcome, assignment.agent_id


def _review_task(
    cand,
    reviewer: Any,
    assignment,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """单个候选人的复核任务闭包：返回 (review_dict, meta, agent_id)。

    只读 cand.match_result，绝不写回（降级不改分）。reviewer 若无 review_traced
    接口，则退化调用 review() 并构造最小 meta，保证兼容。
    """
    if hasattr(reviewer, "review_traced"):
        review_dict, meta = reviewer.review_traced(cand)
        return review_dict, meta, assignment.agent_id

    # 兼容仅提供 review() 的 reviewer（如 DeterministicReviewer 或旧实现）。
    review_dict = reviewer.review(cand)
    meta = {
        "reviewer_mode": "deterministic",
        "llm_success": False,
        "reviewer_fallback_used": False,
        "fallback_reason": "reviewer 不支持 review_traced，按确定性处理",
        "llm_request_attempts": 0,
    }
    return review_dict, meta, assignment.agent_id


def run_extraction_batch(
    resumes,
    client: Any,
    requirements,
    *,
    max_retries: int = 1,
    scheduler_max_retries: int = 0,
    batch_id: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    """批量并发抽取全部简历（阶段一）。

    一次 select_agent 选出 extractor Agent（批次级），再按 shard_size 分片，
    每片独立调用 run_roles_concurrently，结果统一汇总。抽取失败的简历：
    - 保留 CandidateResult（match_result=None）。
    - 生成 ExtractionError，进入返回的 extract_errors。
    - 不进入 Matching Engine / Reviewer / 排名。

    返回：
        {
            "candidates": [CandidateResult, ...],      # 成功与失败都包含
            "extract_errors": [ExtractionError, ...],  # 抽取失败简历
            "trace_records": [dict, ...],              # 阶段一 Trace
            "agent_id": str,                           # 选型结果（供 Trace/调试）
            "agent_name": str,
        }
    """
    from screening_extractors import ExtractionError
    from screening_extraction import ExtractionOutcome
    from screening_orchestrator import CandidateResult

    assignment = select_agent(
        "extractor", context=_job_context(requirements), has_evidence=False
    )

    candidates: dict[str, CandidateResult] = {}
    extract_errors: list[ExtractionError] = []
    trace_records: list[dict[str, Any]] = []
    # 类型安全数据通道：task_key -> 真实 LLM 请求次数，失败场景精确取回（P1 修复）。
    attempts_by_key: dict[str, int] = {}

    keys = [f"{STAGE_EXTRACTION}:{r.resume_id}" for r in resumes]
    by_key = {f"{STAGE_EXTRACTION}:{r.resume_id}": r for r in resumes}

    for shard in _shard_keys(keys, shard_size):
        roles: dict[str, Callable[[], Any]] = {}
        fallbacks: dict[str, Any] = {}
        for task_key in shard:
            resume = by_key[task_key]
            roles[task_key] = (
                lambda r=resume, k=task_key: _extraction_task(
                    r, client, assignment, max_retries, attempts_by_key
                )
            )
            # fallback 的 key 与 roles 完全一致；value 为占位 outcome，
            # 真实错误原因以 RoleResult.error 为准（见下）。
            fallbacks[task_key] = ExtractionOutcome(
                resume_id=resume.resume_id, profile=None, ok=False,
                error="抽取失败（Scheduler 回退）",
            )

        role_results = run_roles_concurrently(
            roles, fallbacks=fallbacks, max_retries=scheduler_max_retries
        )

        for task_key, rr in role_results.items():
            resume = by_key[task_key]
            rid = resume.resume_id

            if rr.ok:
                outcome, agent_id = rr.value
                cand = CandidateResult(
                    resume_id=rid,
                    source_name=resume.source_name,
                    profile=outcome.profile,
                    match_result=None,
                    raw_text=resume.raw_text,
                    explicit_not_owned_skills=outcome.explicit_not_owned_skills,
                    explicit_not_owned_certifications=outcome.explicit_not_owned_certifications,
                    warnings=outcome.warnings,
                )
                candidates[rid] = cand
                trace_records.append(_make_trace_record(
                    task_key=task_key, rid=rid, stage=STAGE_EXTRACTION,
                    agent_role="extractor", agent_id=agent_id,
                    batch_id=batch_id, rr=rr,
                    business_status="success",
                    llm_request_attempts=outcome.llm_request_attempts,
                    scheduler_fallback_used=False,
                    reviewer_fallback_used=False,
                    fallback_reason=None,
                ))
            else:
                # 失败：真实原因在 rr.error（含 ExtractionFailedError 的 outcome.error）。
                error_text = rr.error or "结构化抽取失败"
                candidates[rid] = CandidateResult(
                    resume_id=rid,
                    source_name=resume.source_name,
                    match_result=None,
                    profile=None,
                    raw_text=resume.raw_text,
                    error=error_text,
                )
                extract_errors.append(ExtractionError(
                    source_name=resume.source_name,
                    error=error_text,
                    resume_id=rid,
                ))
                trace_records.append(_make_trace_record(
                    task_key=task_key, rid=rid, stage=STAGE_EXTRACTION,
                    agent_role="extractor", agent_id=assignment.agent_id,
                    batch_id=batch_id, rr=rr,
                    business_status="failed",
                    llm_request_attempts=attempts_by_key.get(rid, 0),
                    scheduler_fallback_used=True,
                    reviewer_fallback_used=False,
                    fallback_reason=error_text,
                ))

    # 保持输入顺序返回 candidates。
    ordered = [candidates[r.resume_id] for r in resumes if r.resume_id in candidates]
    return {
        "candidates": ordered,
        "extract_errors": extract_errors,
        "trace_records": trace_records,
        "agent_id": assignment.agent_id,
        "agent_name": assignment.agent_name,
    }


def run_review_batch(
    candidates,
    reviewer: Any,
    requirements,
    *,
    scheduler_max_retries: int = 0,
    shard_size: int = DEFAULT_SHARD_SIZE,
    batch_id: str = "",
) -> list[dict[str, Any]]:
    """批量并发复核拥有 MatchResult 的候选人（阶段三）。

    一次 select_agent 选出 screening_reviewer Agent（批次级），分片并发。
    只处理 ``cand.match_result is not None`` 的候选人。Reviewer 只读评分，
    降级不改分。确定性降级自身也失败时，保留原始 MatchResult 并记录复核失败。

    返回 trace_records（阶段三 Trace）。cand.review 在任务内部写回。
    """
    assignment = select_agent(
        "screening_reviewer", context=_job_context(requirements), has_evidence=True
    )

    eligible = [c for c in candidates if getattr(c, "match_result", None) is not None]
    keys = [f"{STAGE_REVIEW}:{c.resume_id}" for c in eligible]
    by_key = {f"{STAGE_REVIEW}:{c.resume_id}": c for c in eligible}

    trace_records: list[dict[str, Any]] = []

    for shard in _shard_keys(keys, shard_size):
        roles: dict[str, Callable[[], Any]] = {}
        fallbacks: dict[str, Any] = {}
        for task_key in shard:
            cand = by_key[task_key]
            roles[task_key] = lambda c=cand: _review_task(c, reviewer, assignment)
            # 复核失败的安全空值：明确 needs_human_review，不改评分。
            fallbacks[task_key] = {
                "needs_human_review": True,
                "flags": ["复核失败（Scheduler 回退）"],
                "summary": "复核未能完成，需人工检查。",
            }

        role_results = run_roles_concurrently(
            roles, fallbacks=fallbacks, max_retries=scheduler_max_retries
        )

        for task_key, rr in role_results.items():
            cand = by_key[task_key]
            rid = cand.resume_id

            if rr.ok:
                review_dict, meta, agent_id = rr.value
                cand.review = review_dict
                business_status = "degraded" if meta.get("reviewer_fallback_used") else "success"
                trace_records.append(_make_trace_record(
                    task_key=task_key, rid=rid, stage=STAGE_REVIEW,
                    agent_role="screening_reviewer", agent_id=agent_id,
                    batch_id=batch_id, rr=rr,
                    business_status=business_status,
                    llm_request_attempts=meta.get("llm_request_attempts", 0),
                    scheduler_fallback_used=False,
                    reviewer_fallback_used=bool(meta.get("reviewer_fallback_used")),
                    fallback_reason=meta.get("fallback_reason"),
                ))
            else:
                # Reviewer 任务彻底失败（含确定性降级也失败）：
                # 保留原始 match_result，仅记录复核失败。
                error_text = rr.error or "复核失败"
                cand.review = {
                    "needs_human_review": True,
                    "flags": ["复核失败：" + error_text],
                    "summary": "复核未能完成，评分结果保留。",
                }
                trace_records.append(_make_trace_record(
                    task_key=task_key, rid=rid, stage=STAGE_REVIEW,
                    agent_role="screening_reviewer", agent_id=assignment.agent_id,
                    batch_id=batch_id, rr=rr,
                    business_status="degraded",
                    llm_request_attempts=0,
                    scheduler_fallback_used=True,
                    reviewer_fallback_used=True,
                    fallback_reason=error_text,
                ))

    return trace_records


def _make_trace_record(
    *,
    task_key: str,
    rid: str,
    stage: str,
    agent_role: str,
    agent_id: str,
    batch_id: str,
    rr: Any,
    business_status: str,
    llm_request_attempts: int,
    scheduler_fallback_used: bool,
    reviewer_fallback_used: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    """把一条 RoleResult 转写为 Trace 记录（兼容 trace_view.render_table 外壳）。

    role 字段保存 task_key（兼容旧 Trace 的 role 语义）；agent_role 独立保存
    语义角色；agent_id 保存 Registry 真实 ID。status 映射沿用 core._trace_status
    的取值域 {success, failed, fallback}，供 render_table 展示。
    """
    status = "failed" if not rr.ok else "success"
    if not rr.ok and scheduler_fallback_used:
        status = "failed"
    # Reviewer 降级：Scheduler 成功但业务 degraded，render_table 用 status 表达。
    if rr.ok and business_status == "degraded":
        status = "fallback"

    return {
        # render_table 兼容字段
        "role": task_key,
        "status": status,
        "duration_ms": rr.duration_ms,
        "retries": max(0, rr.attempts - 1),
        "detail": fallback_reason or ("成功" if rr.ok else rr.error),
        "error": None if rr.ok else rr.error,
        # 新增追踪字段
        "batch_id": batch_id,
        "resume_id": rid,
        "task_key": task_key,
        "agent_role": agent_role,
        "agent_id": agent_id,
        "stage": stage,
        "scheduler_status": "success" if rr.ok else "failed",
        "business_status": business_status,
        "scheduler_attempts": rr.attempts,
        "llm_request_attempts": llm_request_attempts,
        "scheduler_fallback_used": scheduler_fallback_used,
        "reviewer_fallback_used": reviewer_fallback_used,
        "fallback_reason": fallback_reason,
    }


def persist_screening_trace(
    trace_records: list[dict[str, Any]],
    *,
    task_id: str,
    mode: str,
    created_at: str,
    runs_dir: Path = RUNS_DIR,
) -> Path | None:
    """把本批次的完整 Trace 写入 runs/ 目录（复用 core.RUNS_DIR 约定）。

    顶层结构 {task_id, mode, created_at, roles:[...]}，与 trace_view.render_table
    兼容。写入失败返回 None，绝不影响主流程。
    """
    import json
    from datetime import datetime

    try:
        trace = {
            "task_id": task_id,
            "mode": mode,
            "created_at": created_at,
            "roles": trace_records,
        }
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        path = runs_dir / f"{stamp}_screening.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception:  # noqa: BLE001 - 轨迹写入失败不得影响主流程
        return None
