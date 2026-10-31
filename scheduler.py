from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REGISTRY_PATH = Path(__file__).with_name("agent_registry.json")


@dataclass(frozen=True)
class AgentAssignment:
    agent_id: str
    agent_name: str
    role: str
    version: str
    instruction: str
    score: float
    reason: str


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Agent 注册表无法读取：{exc}") from exc
    if not isinstance(data.get("agents"), list) or not data["agents"]:
        raise RuntimeError("Agent 注册表没有可用 Agent。")
    return data


def required_capabilities(role: str, context: str, has_evidence: bool = False) -> set[str]:
    required = {
        "planner": {"planning"},
        "solution": {"structure"},
        "reviewer": {"structure"},
        "reviser": {"revision"},
    }.get(role)
    if required is None:
        raise ValueError(f"未知任务角色：{role}")
    if role in {"planner", "reviewer"} and has_evidence:
        required.add("evidence")
    if role in {"solution", "reviewer"} and any(word in context for word in ("指标", "效果", "测试", "数据", "评估")):
        required.add("metrics")
    if role == "solution" and any(word in context for word in ("产品", "服务", "功能", "流程")):
        required.add("product")
    return required


def select_agent(role: str, context: str, has_evidence: bool = False,
                 registry: dict[str, Any] | None = None) -> AgentAssignment:
    registry = registry or load_registry()
    required = required_capabilities(role, context, has_evidence)
    candidates = [a for a in registry["agents"] if a.get("enabled", True) and a.get("role") == role]
    if not candidates:
        raise RuntimeError(f"没有启用的 {role} Agent。")
    scored = []
    for agent in candidates:
        caps = set(agent.get("capabilities", []))
        coverage = len(required & caps) / len(required)
        load = max(0.0, min(1.0, float(agent.get("load", 0))))
        expertise = max(0.0, min(1.0, float(agent.get("expertise", 0.5))))
        priority = float(agent.get("priority", 0))
        score = coverage * 100 + expertise * 10 + priority - load * 20
        scored.append((score, agent, coverage))
    score, chosen, coverage = max(scored, key=lambda row: (row[0], row[1].get("id", "")))
    missing = sorted(required - set(chosen.get("capabilities", [])))
    reason = f"需求能力：{', '.join(sorted(required))}；能力覆盖 {coverage:.0%}；负载 {float(chosen.get('load', 0)):.0%}"
    if missing:
        reason += f"；未覆盖：{', '.join(missing)}"
    return AgentAssignment(chosen["id"], chosen.get("name", chosen["id"]), role,
                           str(registry.get("version", "1")), chosen.get("instruction", ""), score, reason)


@dataclass
class RoleResult:
    """单个角色在一次编排运行中的最终结果。

    只有当角色在某次尝试成功时 `ok` 才为 True；重试耗尽则 `ok=False`，
    并返回 `value`（调用方提供的安全空值/回退值）与 `error` 记录失败原因，
    绝不把失败伪装成成功。
    """
    role: str
    value: Any
    ok: bool
    attempts: int
    error: str | None
    duration_ms: int
    failures: tuple[str, ...] = ()


def run_roles_concurrently(
    roles: dict[str, Callable[[], Any]],
    fallbacks: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> dict[str, RoleResult]:
    """并行执行一组彼此独立的角色，并附带回退与重试。

    每个角色在独立线程中运行：首次失败后最多再重试 ``max_retries`` 次
    （总尝试 ``max_retries + 1`` 次）。任一尝试成功即返回该次调用的成功结果；
    重试耗尽则回退到 ``fallbacks`` 中该角色的安全值（未提供时为 ``None``），
    每次失败都会写入 ``RoleResult.failures``（包含尝试序号和异常原因）；
    ``RoleResult.error`` 保留最后一次异常以兼容已有调用方。重试耗尽时
    回退到安全值并明确标记失败。

    各角色共享同一份不可变输入，彼此不读取对方输出，因此可以安全并行；
    ``max_workers`` 取角色数，使所有角色真正并发。返回字典保持输入顺序，
    调用方按角色名取值即可。
    """
    if max_retries < 0:
        raise ValueError("max_retries 不能小于 0。")
    if not roles:
        return {}
    fallbacks = fallbacks or {}

    def _run_one(role: str, call: Callable[[], Any]) -> RoleResult:
        started = time.perf_counter()
        last_error: str | None = None
        failures: list[str] = []
        for attempt in range(1, max_retries + 2):
            try:
                value = call()
            except Exception as exc:  # noqa: BLE001 - 记录任意异常原因，供回退与测试断言
                last_error = f"{type(exc).__name__}: {exc}"
                failures.append(
                    f"第 {attempt}/{max_retries + 1} 次尝试异常：{last_error}"
                )
                continue
            return RoleResult(role=role, value=value, ok=True, attempts=attempt,
                              error=None, duration_ms=int((time.perf_counter() - started) * 1000),
                              failures=tuple(failures))
        # 重试耗尽：回退到安全值，明确失败，不伪装成通过。
        return RoleResult(role=role, value=fallbacks.get(role), ok=False,
                          attempts=max_retries + 1, error=last_error,
                          duration_ms=int((time.perf_counter() - started) * 1000),
                          failures=tuple(failures))

    results: dict[str, RoleResult] = {}
    with ThreadPoolExecutor(max_workers=len(roles)) as pool:
        futures = {pool.submit(_run_one, role, call): role for role, call in roles.items()}
        for future in as_completed(futures):
            result = future.result()
            results[result.role] = result
    # 按输入顺序返回，保证调用方取值稳定。
    return {role: results[role] for role in roles}
