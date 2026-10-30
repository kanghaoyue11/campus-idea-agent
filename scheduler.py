from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
