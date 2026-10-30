from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from scheduler import load_registry, select_agent


APP_NAME = "校园创意验证 Agent"
MAX_RETRIES = 2
KNOWLEDGE_PATH = Path(__file__).with_name("knowledge_base.json")


def retrieve_knowledge(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Retrieve local guidance by curated keywords; this is lexical RAG, not vector search."""
    try:
        records = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scored = []
    for record in records:
        score = sum(1 for keyword in record.get("keywords", []) if keyword and keyword in query)
        if score:
            scored.append((score, record))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("id", "")))
    return [{"id": r.get("id", ""), "title": r.get("title", ""), "content": r.get("content", "")}
            for _, r in scored[:limit]]


def _knowledge_context(records: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{r['title']}]\n{r['content']}" for r in records) or "未检索到相关本地知识条目。"


@dataclass
class RunLog:
    step: str
    status: str
    detail: str
    duration_ms: int


class LLMClient:
    def __init__(self, mode: str, base_url: str, model: str, api_key: str):
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(self, system: str, user: str) -> str:
        if self.mode == "模拟模式（不调用真实模型）":
            return ""
        if not self.api_key:
            raise ValueError("真实模型模式需要 API Key。请设置 OPENAI_API_KEY 环境变量后重新启动。")
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"模型接口返回 HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"无法连接模型接口：{exc}") from exc
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"模型接口返回格式无法识别：{exc}") from exc


def mock_plan(problem: str, audience: str, evidence: str) -> str:
    evidence_line = evidence.strip() if evidence.strip() else "尚无一手证据，先通过访谈收集。"
    return f"""# 本地模拟验证计划

- 验证目标：确认“{audience}”是否遇到“{problem}”，并了解他们目前如何处理。
- 当前证据：{evidence_line}
- 待验证假设：目标用户最近遇到过该问题；现有办法存在明显阻碍；用户愿意尝试更省步骤的解决方式。

## 三天行动
1. 第一天：招募5位目标用户，分别询问最近一次遇到该问题的具体过程。
2. 第二天：整理共同阻碍，做一个可人工交付的页面或服务草图。
3. 第三天：请至少3位用户完成一个具体任务，记录是否完成和主要困难。

## 访谈问题
- 你最近一次遇到这个问题是什么时候？当时怎么解决？
- 哪一步最费时间或最容易出错？
- 你现在会尝试什么替代办法？

## 观察指标与风险
记录访谈人数、任务完成情况和用户原话；这些是计划指标，不能提前当作结果。样本较少时只用于早期判断，并避免收集不必要的个人信息。"""


def mock_solution(problem: str, audience: str, evidence: str, plan: str) -> str:
    return f"""# 项目方案草案：校园创意验证

## 一句话介绍
帮助{audience}更低成本地解决“{problem}”，先用小规模验证决定是否继续投入。

## 用户痛点与证据
- 当前输入的需求：{problem}
- 用户群体：{audience}
- 已提供证据：{evidence if evidence else '尚无一手证据；本项为待验证假设。'}
- 关键提醒：没有访谈、问卷或观察数据前，不将需求规模和付费意愿表述为事实。

## 验证计划
{plan}

## 最小可行方案
先用可人工交付的服务原型完成5位目标用户测试，记录任务完成率、满意度和再次使用意愿。验证通过后再开发自动化功能。

## 成效指标（目标值，需验证）
- 完成至少5次目标用户访谈
- 至少3位用户愿意尝试原型
- 记录每位测试者的任务完成情况与主要反馈

## 风险与边界
样本少不能代表整个校园；不收集不必要的个人敏感信息；模型生成内容须由参赛者核实。"""


def mock_review(problem: str, audience: str, evidence: str, solution: str) -> dict[str, Any]:
    flags = []
    if not problem.strip():
        flags.append("需求描述为空")
    if not audience.strip():
        flags.append("目标用户未明确")
    if not evidence.strip():
        flags.append("缺少一手证据，文中已提示待验证")
    required = ["用户痛点与证据", "验证计划", "成效指标", "风险与边界"]
    missing = [section for section in required if section not in solution]
    if missing:
        flags.append("方案缺少章节：" + "、".join(missing))
    return {"passed": len(missing) == 0 and bool(problem.strip()) and bool(audience.strip()),
            "issues": flags, "missing_sections": missing,
            "summary": "检查完成：已核对必需章节、用户信息与证据边界。"}


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("验证 Agent 没有返回有效 JSON。")
    return json.loads(text[start:end + 1])


def run_orchestration(problem: str, audience: str, evidence: str, client: LLMClient) -> dict[str, Any]:
    logs: list[RunLog] = []
    started = time.perf_counter()
    knowledge = retrieve_knowledge(f"{problem} {audience} {evidence}")
    knowledge_context = _knowledge_context(knowledge)
    logs.append(RunLog("本地知识检索", "完成", f"检索到 {len(knowledge)} 条本地指南", 0))
    registry = load_registry()
    plan_assignment = select_agent("planner", problem, bool(evidence.strip()), registry)
    solution_assignment = select_agent("solution", problem, bool(evidence.strip()), registry)
    reviewer_assignment = select_agent("reviewer", problem, bool(evidence.strip()), registry)
    reviser_assignment = select_agent("reviser", problem, bool(evidence.strip()), registry)
    assignments = [plan_assignment, solution_assignment, reviewer_assignment, reviser_assignment]
    dispatch_status = "模拟派发" if client.mode.startswith("模拟") else "已派发"
    for assignment in assignments:
        logs.append(RunLog("动态任务调度", dispatch_status,
                           f"{assignment.agent_name} v{assignment.version}｜{assignment.reason}", 0))
    if client.mode.startswith("模拟模式"):
        plan = mock_plan(problem, audience, evidence)
        logs.append(RunLog("规划 Agent", "完成", "拆解目标、假设和验证步骤（模拟）", 1))
        solution = mock_solution(problem, audience, evidence, plan)
        logs.append(RunLog("方案 Agent", "完成", "生成方案初稿（模拟）", 1))
        review = mock_review(problem, audience, evidence, solution)
        logs.append(RunLog("独立验证 Agent", "通过" if review["passed"] else "需修改", review["summary"], 1))
        attempts = 0
    else:
        plan = client.complete(
            f"你是规划 Agent（当前分派：{plan_assignment.agent_name}）。{plan_assignment.instruction}只输出清晰、可执行的校园需求验证计划，区分已知信息和待验证假设。可以参考提供的本地指南，但不得把指南说成用户证据。",
            f"本地指南：\n{knowledge_context}\n需求：{problem}\n目标用户：{audience}\n用户提供的证据：{evidence or '无'}",
        )
        logs.append(RunLog("规划 Agent", "完成", "输出验证步骤与假设", 0))
        solution = client.complete(
            f"你是方案 Agent（当前分派：{solution_assignment.agent_name}）。{solution_assignment.instruction}基于输入写中文项目方案草案。必须包含：一句话介绍、用户痛点与证据、验证计划、最小可行方案、成效指标、风险与边界。不得编造调研数据。",
            f"本地指南：\n{knowledge_context}\n需求：{problem}\n目标用户：{audience}\n已有证据：{evidence or '无；明确写待验证'}\n验证计划：\n{plan}",
        )
        logs.append(RunLog("方案 Agent", "完成", "生成方案初稿", 0))
        review = {}
        attempts = 0
        while attempts <= MAX_RETRIES:
            check_started = time.perf_counter()
            review_text = client.complete(
                f"你是独立验证 Agent（当前分派：{reviewer_assignment.agent_name}）。{reviewer_assignment.instruction}检查方案是否包含：用户痛点与证据、验证计划、成效指标、风险与边界；是否将无证据推测说成事实。只返回 JSON：{{\"passed\":boolean,\"issues\":[string],\"missing_sections\":[string],\"summary\":string}}",
                f"用户需求：{problem}\n目标用户：{audience}\n用户证据：{evidence or '无'}\n方案：\n{solution}",
            )
            review = parse_json(review_text)
            logs.append(RunLog("独立验证 Agent", "通过" if review.get("passed") else "需修改", str(review.get("summary", "检查完成")), int((time.perf_counter()-check_started)*1000)))
            if review.get("passed") or attempts == MAX_RETRIES:
                break
            attempts += 1
            solution = client.complete(
                f"你是方案修订 Agent（当前分派：{reviser_assignment.agent_name}）。{reviser_assignment.instruction}根据验证问题修订原方案，保留真实信息，不得补造数据。输出完整修订版。",
                f"验证问题：{review.get('issues', [])}\n原方案：\n{solution}\n需求：{problem}\n用户：{audience}\n证据：{evidence or '无'}",
            )
            logs.append(RunLog("Self-Loop 修订", f"第 {attempts} 次", "根据独立验证结果修订方案", 0))
    logs.append(RunLog("流程结束", "完成", f"总耗时约 {int((time.perf_counter()-started)*1000)} ms；最大返工次数 {MAX_RETRIES}", 0))
    return {"plan": plan, "solution": solution, "review": review, "attempts": attempts,
            "logs": [asdict(log) for log in logs], "mode": client.mode,
            "knowledge_sources": [{"id": r["id"], "title": r["title"]} for r in knowledge],
            "scheduled_agents": [{"id": a.agent_id, "name": a.agent_name, "role": a.role,
                                  "version": a.version, "reason": a.reason} for a in assignments],
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def to_markdown(result: dict[str, Any]) -> str:
    review = result["review"]
    lines = [f"# {APP_NAME}运行结果", "", f"- 模式：{result['mode']}", f"- 时间：{result['created_at']}",
             f"- Self-Loop 返工次数：{result['attempts']}（上限 {MAX_RETRIES}）", "", "## 验证报告",
             f"- 状态：{'通过' if review.get('passed') else '未通过或需人工复核'}",
             f"- 摘要：{review.get('summary', '')}", "- 问题：" + ("；".join(review.get("issues", [])) or "无"),
             "", "## 本地知识指南", "；".join(x["title"] for x in result.get("knowledge_sources", [])) or "无匹配条目",
             "", "## 验证计划", result["plan"], "", "## 项目方案", result["solution"], "", "## 运行日志"]
    lines.extend(f"- {x['step']}｜{x['status']}｜{x['detail']}" for x in result["logs"])
    return "\n".join(lines)


