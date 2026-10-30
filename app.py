from __future__ import annotations

import json
import os
import streamlit as st

from core import (APP_NAME, MAX_RETRIES, LLMClient, run_orchestration, to_markdown)

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")
st.title("🧭 校园创意验证 Agent")
st.caption("多 Agent 编排演示：规划 → 方案生成 → 独立验证 → 有上限的 Self-Loop 修订")
with st.sidebar:
    st.header("模型设置")
    mode = st.radio("运行方式", ["模拟模式（不调用真实模型）", "真实模型 API"], help="模拟模式的输出来自本地固定逻辑，不代表真实模型能力。")
    base_url = st.text_input("兼容接口地址", value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model = st.text_input("模型名称", value=os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    if mode.startswith("真实"):
        st.info("API Key 只从 OPENAI_API_KEY 环境变量读取，不会写入项目文件。")
    st.caption(f"单次运行最多返工 {MAX_RETRIES} 次。")

st.subheader("描述一个校园真实需求")
problem = st.text_area("遇到的具体问题", placeholder="例如：新生常常不知道去哪办理校园卡补办，需要在多个网页和群聊里找信息。", height=100)
audience = st.text_input("主要服务谁", placeholder="例如：刚入学的新生")
evidence = st.text_area("你已经观察或收集到的证据（可选）", placeholder="例如：我问过3位同学，其中2位说最近一周找过相关信息。没有证据就留空。", height=80)
if st.button("运行 Agent 编排", type="primary", use_container_width=True):
    if not problem.strip() or not audience.strip():
        st.error("请先填写需求和目标用户。")
    else:
        client = LLMClient(mode, base_url, model, os.getenv("OPENAI_API_KEY", ""))
        with st.spinner("Agent 正在按流程处理…"):
            try:
                st.session_state["last_result"] = run_orchestration(problem.strip(), audience.strip(), evidence.strip(), client)
            except Exception as exc:
                st.error(f"运行失败：{exc}")

result = st.session_state.get("last_result")
if result:
    st.divider()
    if result["mode"].startswith("模拟"):
        st.warning("当前结果来自模拟模式，不是真实模型生成。切换到真实模型 API 后，才能验证真实模型接入效果。")
    else:
        st.success("本次运行调用了配置的真实模型 API。请人工复核生成内容。")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 规划 Agent 输出")
        st.markdown(result["plan"])
    with c2:
        st.markdown("### 独立验证报告")
        st.json(result["review"])
        st.metric("Self-Loop 修订次数", result["attempts"], f"上限 {MAX_RETRIES}")
    st.markdown("### 动态 Agent 派发")
    st.dataframe(result.get("scheduled_agents", []), use_container_width=True, hide_index=True)
    st.markdown("### 最终方案")
    st.markdown(result["solution"])
    st.markdown("### 可观测运行日志")
    st.dataframe(result["logs"], use_container_width=True, hide_index=True)
    st.markdown("### 本地知识检索")
    st.caption("检索采用关键词匹配本地指南，不联网；它为方案提供方法参考，不构成用户证据。")
    st.write("、".join(item["title"] for item in result.get("knowledge_sources", [])) or "本次没有匹配到指南。")
    markdown = to_markdown(result)
    st.download_button("下载 Markdown 运行报告", markdown, file_name="campus_idea_agent_report.md", mime="text/markdown")
    st.download_button("下载 JSON 运行数据", json.dumps(result, ensure_ascii=False, indent=2), file_name="campus_idea_agent_run.json", mime="application/json")

st.divider()
st.caption("安全边界：不要输入身份证号、联系方式等敏感信息；方案里的目标值是待验证指标，不是已经取得的成果。")
