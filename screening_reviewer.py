"""Reviewer Agent（招聘筛选流水线第五阶段）。

对 Matching Engine 的评分结果生成独立复核建议。Reviewer 只提出复核意见，
绝不修改 Matching Engine 的评分、推荐等级或候选人排序，最终保留人工决定权。

复核重点：
    - Evidence 是否充分（是否存在无证据支撑的判定）；
    - 是否存在信息缺失（unknown_items / information_completeness）；
    - 是否存在证据矛盾（positive 与 negative 冲突已由 Matching Engine 降级 unknown，
      这里提示人工关注）；
    - 是否存在抽取异常（warnings / error）；
    - 是否需要人工复核（needs_human_review）。

提供两种实现：
    - DeterministicReviewer：纯规则，不调用 LLM，输出确定性的复核意见。
    - LLMReviewer：可选，调用 LLM 生成自然语言意见（仍不改评分）。
"""

from __future__ import annotations

from typing import Any

from screening_models import MatchResult


class DeterministicReviewer:
    """纯规则复核建议生成器（不调用 LLM，可复现）。"""

    def review(self, cand: Any) -> dict[str, Any]:
        """根据候选人结果生成复核建议 dict。

        cand 需至少提供 match_result / profile / warnings / error 属性
        （来自 screening_orchestrator.CandidateResult）。
        """
        result: MatchResult | None = cand.match_result
        if result is None:
            return {
                "needs_human_review": True,
                "flags": [f"抽取失败，无法评分：{cand.error or '未知原因'}"],
                "summary": "该简历未能完成结构化抽取，需人工检查。",
            }

        flags: list[str] = []
        # 信息缺失
        if result.information_completeness is not None and result.information_completeness < 100.0:
            missing = result.unknown_items or ()
            flags.append(
                f"存在信息缺失（完整度 {result.information_completeness}%），未判定项："
                + ("；".join(missing) if missing else "见 unknown_items")
            )
        # 明确不满足
        if result.not_matched_items:
            flags.append("明确不满足项：" + "；".join(result.not_matched_items))
        # 证据充分性
        if not result.evidence:
            flags.append("没有任何有效 Evidence 支撑本次评分。")
        # 抽取异常
        if getattr(cand, "warnings", ()):
            flags.append("抽取告警：" + "；".join(cand.warnings))
        if getattr(cand, "error", None):
            flags.append("抽取异常：" + cand.error)

        needs_review = bool(flags) or result.needs_human_review
        if flags:
            summary = "建议人工复核：" + "；".join(flags[:3]) + ("…" if len(flags) > 3 else "")
        else:
            # 无告警项，但招聘结果一律保留人工决定权（needs_human_review 恒 True）。
            summary = "未发现需额外关注的问题；按流程仍保留人工最终决定权。"
        return {
            "needs_human_review": needs_review,
            "flags": flags,
            "summary": summary,
        }


class LLMReviewer:
    """调用 LLM 生成复核意见（不改评分、推荐等级、排序）。"""

    def __init__(self, client: Any):
        self.client = client

    def review(self, cand: Any) -> dict[str, Any]:
        """兼容旧调用者的公共接口：返回复核建议 dict。

        内部委托 review_traced，仅返回 dict，丢弃 meta。行为与改造前一致。
        """
        review_dict, _meta = self.review_traced(cand)
        return review_dict

    def review_traced(self, cand: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """返回 (review_dict, meta)，meta 用于追踪降级与请求次数。

        meta 字段：
            reviewer_mode          "llm" | "deterministic"
            llm_success            是否真正完成了 LLM 复核
            reviewer_fallback_used 是否发生了降级
            fallback_reason        降级原因（None 表示未降级）
            llm_request_attempts   实际调用 client.complete() 的次数（0 或 1）
        降级成功绝不改变 cand.match_result（本方法只读 match_result）。
        """
        result: MatchResult | None = cand.match_result
        if result is None:
            # 无评分结果：直接确定性意见，未调用 LLM。
            meta = {
                "reviewer_mode": "deterministic",
                "llm_success": False,
                "reviewer_fallback_used": False,
                "fallback_reason": "match_result 为空，无法调用 LLM",
                "llm_request_attempts": 0,
            }
            return DeterministicReviewer().review(cand), meta

        system = (
            "你是招聘筛选复核 Agent。你的职责是检查评分结果的证据充分性、"
            "信息缺失、证据矛盾与抽取异常，并给出人工复核建议。"
            "你只能提建议，不得修改任何分数、推荐等级或排序。只输出 JSON："
            '{"needs_human_review":boolean,"flags":[string],"summary":string}'
        )
        payload = {
            "recommendation": result.recommendation,
            "overall_score": result.overall_score,
            "information_completeness": result.information_completeness,
            "unknown_items": list(result.unknown_items),
            "not_matched_items": list(result.not_matched_items),
            "evidence_count": len(result.evidence),
        }
        import json as _json
        user = "评分结果：\n" + _json.dumps(payload, ensure_ascii=False)
        try:
            text = self.client.complete(system, user)
            from core import parse_json
            data = parse_json(text)
            data.setdefault("needs_human_review", True)
            data.setdefault("flags", [])
            data.setdefault("summary", "复核完成。")
            meta = {
                "reviewer_mode": "llm",
                "llm_success": True,
                "reviewer_fallback_used": False,
                "fallback_reason": None,
                "llm_request_attempts": 1,
            }
            return data, meta
        except Exception as exc:  # noqa: BLE001 - 复核失败回退确定性意见，不影响主流程
            fallback = DeterministicReviewer().review(cand)
            meta = {
                "reviewer_mode": "deterministic",
                "llm_success": False,
                "reviewer_fallback_used": True,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
                "llm_request_attempts": 1,
            }
            return fallback, meta
