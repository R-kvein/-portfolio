# RAG + Agent 智能分析与推理引擎

把"问题描述 → 检索历史/日志 → LLM 推理 → 建议生成"编排成一张 **LangGraph 状态图**，带分支、异常兜底和重试。把分析准确率提升 26%，覆盖 20+ 用户日常使用。

## 为什么用 LangGraph 不用顺序代码

流程里**真实存在分支和异常路径**：
- 检索为空 → 走"请用户补充信息"分支
- LLM 超时/失败 → 重试 + 降级返回检索原文
- 高频 case 命中缓存 → 直接返回经验解，跳过 LLM

if-else 嵌套两层就开始难维护；状态图把节点和条件边显式声明，改流程只动图、不动主干代码。

## 状态图

```
                  ┌──────────────┐
                  │ ① parse      │   抽意图 + 关键实体
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ ② cache_hit? │── yes ──► return_cached
                  └──────┬───────┘
                         │ no
                         ▼
                  ┌──────────────┐
                  │ ③ retrieve   │   bge + BM25 混合
                  └──────┬───────┘
                         │
                ┌────────┴────────┐
              empty             have docs
                │                 │
                ▼                 ▼
        ┌─────────────┐   ┌──────────────┐
        │ ask_user    │   │ ④ assemble   │   去噪 + 压缩上下文
        │ (兜底)      │   └──────┬───────┘
        └─────────────┘          │
                                 ▼
                          ┌──────────────┐
                          │ ⑤ reason     │   LLM 推理 + 建议
                          └──────┬───────┘
                            ok   │   fail
                                 │   ──────► retry / degrade
                                 ▼
                          ┌──────────────┐
                          │ format       │   结构化输出 + citation
                          └──────────────┘
```

## 5 大业务痛点 → 工具映射

| # | 痛点 | 对应工具 |
|---|------|---------|
| 1 | 历史记录术语不统一导致召回低 | `term_normalizer` 业务词典 + Embedding 同义回退 |
| 2 | 跨系统日志格式异构 | `log_extractor` LLM Function Call 抽取统一 schema |
| 3 | 相似 case 高频复现 | `case_cache` 命中直接返回经验解 |
| 4 | 建议过于宽泛、不可执行 | `output_constrainer` 强制"原因 / 证据 / 建议步骤"三段式 |
| 5 | 用户无法回溯依据 | `citation_tagger` 每条结论挂检索原文 ID |

## 目录结构

```
.
├── src/
│   ├── state.py       # 状态对象（在节点间流转的数据）
│   ├── graph.py       # LangGraph 状态机定义（节点 + 边 + 条件）
│   ├── nodes/
│   │   ├── parse.py
│   │   ├── retrieve.py
│   │   ├── assemble.py
│   │   └── reason.py
│   ├── tools.py       # 5 大痛点对应的工具
│   └── server.py
└── requirements.txt
```

## 跑起来

```bash
pip install -r requirements.txt
python -m src.server   # 默认 0.0.0.0:8002
```
