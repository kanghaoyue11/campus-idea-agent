# 有据：AI 简历筛选助手

“有据”是一款本地运行的招聘简历批量筛选原型。它将简历事实抽取、原文证据检查、确定性评分、候选人排序和人工复核提示组织成可追踪的处理流程。

## 设计原则

- 模型只负责把简历内容整理成结构化信息，并给出待核查提示；分数、推荐等级与排序由确定性规则引擎计算。
- 结构化判断附带简历原文证据。引文无法在原文定位时，不将它作为有效支持。
- 简历未提及某项信息时标记为未知或待核实，不直接当作不符合。
- 任何结果均由招聘人员复核；系统不自动录用或淘汰候选人。
- 模拟模式只演示本地流程，不代表真实模型的准确率、速度或招聘效果。

## 流程

岗位标准设置 → PDF/TXT 文本提取 → 批量结构化抽取 → 原文证据校验 → 确定性匹配评分 → 批量复核提示 → 排序和运行记录。

抽取与复核是有依赖关系的阶段，依次执行；每一阶段中互相独立的简历任务使用标准库线程池并发处理。调度过程记录任务状态、错误、重试和耗时。

## 本地运行

需要 Python 3.10 或更新版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run screening_app.py --server.address 127.0.0.1 --server.port 8501
```

默认可选择模拟模式，不需要 API Key。Windows 用户可按相同步骤在虚拟环境中安装依赖并运行 Streamlit。

## 真实模型配置（可选）

真实模型模式会把必要的简历文本发送到使用者配置的模型服务商。先了解服务商的数据处理条款并获得相应授权；演示时优先使用合成简历。

```dotenv
OPENAI_API_KEY=在本机设置，不要提交到 GitHub
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus
```

将变量仅保存在本机环境或被 Git 忽略的 `.env` 文件中。切勿在仓库、截图、视频或聊天中公开密钥。

## 主要模块

| 模块 | 职责 |
|---|---|
| `screening_app.py` | 简历上传、岗位设置和结果界面 |
| `screening_orchestrator.py` | 组织完整筛选阶段并汇总报告 |
| `screening_scheduler.py` | 批次分片、并发任务和 Trace |
| `screening_extraction.py` | 模型抽取结果清洗与证据检查 |
| `screening_extractors.py` | PDF/TXT 文本解析 |
| `screening_models.py` | 岗位、简历、证据和评分数据结构 |
| `matching.py` | 确定性评分、状态判断与排序 |
| `screening_reviewer.py` | 生成复核提示，不修改评分结果 |
| `core.py` / `scheduler.py` | 模型接口、轨迹基础类型和通用任务派发 |

## 测试

```bash
python -m unittest discover -s . -p 'test_screening_*.py' -v
```

自动化测试主要使用本地替身，不调用真实模型服务。测试通过只说明测试覆盖的代码路径符合断言，不代表真实模型准确、公平或适用于实际招聘。

## 限制与安全

当前版本是本地原型，没有生产级账号权限、云端部署或招聘系统集成。原文引句可以定位文本来源，但不能证明简历陈述属实；技能别名、岗位权重和阈值仍需专业人员审定；尚无真实招聘数据盲测或公平性评估。简历包含个人信息，未经授权不要上传；公开仓库不得包含真实简历、运行 Trace、API Key 或个人信息。分数仅用于辅助复核，不应单独作为淘汰依据。
