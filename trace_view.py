from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).with_name("runs")


def latest_trace(runs_dir: Path = RUNS_DIR) -> dict | None:
    """返回 runs/ 下最新一份轨迹（按文件名时间戳字典序取最大），无则返回 None。"""
    if not runs_dir.is_dir():
        return None
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        return None
    latest = files[-1]
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fmt_ms(ms) -> str:
    try:
        return f"{int(ms)} ms"
    except (TypeError, ValueError):
        return "-"


def render_table(trace: dict) -> str:
    lines: list[str] = []
    lines.append(f"任务 ID：{trace.get('task_id', '-')}")
    lines.append(f"模式：{trace.get('mode', '-')}｜创建时间：{trace.get('created_at', '-')}")
    lines.append("")
    header = ("角色", "状态", "耗时", "重试", "说明")
    widths = [18, 12, 10, 6, 0]  # 最后一列说明自适应，不设固定宽度
    lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    lines.append("  ".join("-" * w for w in widths) + "  " + "-" * 20)
    for role in trace.get("roles", []):
        name = role.get("role", "-")
        status = role.get("status", "-")
        duration = _fmt_ms(role.get("duration_ms"))
        retries = str(role.get("retries", 0))
        detail = (role.get("error") or role.get("detail") or "").replace("\n", " ")
        if len(detail) > 40:
            detail = detail[:40] + "…"
        lines.append(
            f"{name.ljust(18)}  {status.ljust(12)}  {duration.ljust(10)}  {retries.ljust(6)}  {detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    trace = latest_trace()
    if trace is None:
        print("runs/ 下没有可用轨迹。请先运行一次 app.py。")
        return 1
    print(render_table(trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
