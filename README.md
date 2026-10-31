# 校园创意验证 Agent

校园创意验证 Agent 是一个本地运行的多智能体应用，用来把校园服务创意整理成可执行的早期验证计划。使用者填写问题、目标用户和已有证据后，可以查看规划结果、方案草稿、检查状态及运行日志。

## 功能

- 规划 Agent 与方案 Agent 在真实模型模式下读取同一份输入快照，并行运行。
- 独立验证 Agent 检查方案结构与证据表达；真实模型模式下，未通过时最多进行两轮修订。
- 并行角色调用失败后最多重试两次。每次失败单独记录角色、尝试序号和异常原因；重试耗尽时返回明确失败状态和安全空值。
- 按任务能力、角色配置、优先级和静态负载选择 Agent，并记录派发理由。
- 运行轨迹以 JSON 保存角色状态、耗时、异常和重试次数；运行 python trace_view.py 可查看最近的时间线。
- 内置项目验证指南通过本地关键词匹配，不访问外部网站。
- 支持模拟模式和兼容 Chat Completions 的真实模型 API，可下载 Markdown 报告和 JSON 运行记录。

模拟模式使用本地确定性逻辑，不调用模型，也不证明真实 API、并行模型调用或模型效果。真实模型模式才会并行调用规划与方案角色。页面和导出内容会标明运行模式。

## 运行

需要 Python 3.10 或更新版本。

在 macOS 上可双击 启动应用.command，或在项目目录运行：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
~~~

浏览器打开 Streamlit 显示的本地地址。默认选择模拟模式，不需要 API Key。

## 配置真实模型

设置 API Key、兼容接口根地址和模型名称，然后从同一终端启动：

~~~bash
export OPENAI_API_KEY="你的密钥"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="服务商支持的模型名"
streamlit run app.py
~~~

应用会在接口根地址后追加 /chat/completions。不同服务商的鉴权、模型名称和请求格式可能不同。请先查服务商文档，并避免把真实密钥放入代码、截图、报告或仓库。当前仓库不包含 API Key，也没有已记录的真实模型端到端实测结果。

## 项目结构

~~~text
app.py          Streamlit 页面
core.py         编排流程、模型客户端、本地指南检索与轨迹记录
scheduler.py    Agent 注册表、能力派发、并行角色调度和有界重试
agent_registry.json  Agent 角色及静态调度配置
knowledge_base.json  本地项目验证指南
trace_view.py   最近一次运行轨迹查看工具
test_core.py   核心自动化测试
~~~

## 测试

核心用例基于 Python 标准库 unittest，不需要 pytest：

~~~bash
python -m unittest test_core.py -v
~~~

当前本地自动化测试共 13 项通过。用例包括模拟流程、Self-Loop 上限、API 请求组装、缺少密钥提示、失败重试、耗尽状态和使用 Barrier/Event 验证两个并行角色确实重叠。测试使用本地替身，不会调用真实模型服务。

## 当前范围

这是一个单机原型，角色和工作流固定在代码中。规划与方案角色可以并行执行，验证和修订按数据依赖运行。注册表中的负载是静态配置，不是实时资源监控。当前没有通用 DAG 编辑器、远程 Agent 动态注册、MCP 外部工具、外部搜索、向量数据库、持久记忆、用户账号或云端部署。验证结果也不是事实核查。

真实模型性能、并行提速比例和用户试点效果尚未测量。不要将方案中的评测计划当作已取得的结果。

## 项目材料

- 参赛方案书：参赛方案书.pdf
- 方案讲解：校园创意验证Agent_演示稿.pptx
- 录制脚本：演示脚本与视频素材计划.md

公开源码仓库：<https://github.com/kanghaoyue11/campus-idea-agent>
