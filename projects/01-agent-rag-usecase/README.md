# Agent + RAG: 自然语言需求 → 结构化执行用例

LangChain + LangGraph 的 ReAct Agent，把业务方写的自然语言需求文档自动转成结构化执行用例。底层是 LoRA 微调的 Qwen2.5-7B，用 vLLM 推理、FastAPI 对外提供接口。

## 成果

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 用例输出准确率（80 条固定测试集） | 60% | **88%** |
| 端到端处理一份需求 | ~25 min | **~15 min**（-40%） |
| 推理 QPS（vLLM vs transformers 原生） | 1× | **~5×** |

准确率拆解（同一测试集，控制变量）：
- chunk size 400→600 + overlap 100 → 召回率 +12pt
- BM25 + 向量混合检索 → +8pt
- Cross-Encoder rerank（Top-20 → Top-3）→ +5pt
- Few-shot prompt + 结构化约束 → +3pt

## 架构

```
        用户自然语言需求
                │
                ▼
   ┌────────────────────────────┐
   │ ReAct Agent (LangChain)    │◄──┐
   │ Thought → Action → Obs     │   │ 循环
   └────────────┬───────────────┘   │
                │ Tool Calling      │
       ┌────────┼────────┐          │
       ▼        ▼        ▼          │
   history_  usecase_  mcp_       ──┘
   search    generate  internal
   (RAG)     (LoRA)    (内部规则)
                │
                ▼
       LoRA-Qwen2.5 via vLLM
                │
                ▼
       结构化 JSON 用例
       (FastAPI 返回)
```

## 目录结构

```
.
├── src/
│   ├── config.py          # 集中配置（路径、模型、阈值）
│   ├── llm.py             # vLLM 客户端 + LoRA 适配器加载
│   ├── retriever.py       # Chroma + BM25 混合检索 + Rerank
│   ├── tools.py           # 三类 Tool（RAG / 生成 / MCP）
│   ├── agent.py           # ReAct Agent 主循环
│   └── server.py          # FastAPI 接口
├── finetune/
│   ├── lora_config.yaml   # LLaMA-Factory 配置
│   ├── train.sh           # 一键启动 LoRA 训练
│   └── sample_data.jsonl  # 训练数据样例（5 条）
├── eval/
│   ├── evaluate.py        # 准确率评估脚本（60→88 就是它跑出来的）
│   └── testset_sample.jsonl
└── Dockerfile
```

## 跑起来

```bash
# 1. 准备 vLLM 服务（一张 24GB 显卡，加载 LoRA 适配器）
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora --lora-modules usecase=./finetune/output/lora \
  --max-loras 1 --port 8001

# 2. 启动应用
pip install -r requirements.txt
python -m src.server   # FastAPI 默认 0.0.0.0:8000

# 3. 调一下
curl -X POST localhost:8000/usecase \
  -H "Content-Type: application/json" \
  -d '{"requirement": "用户在结算页选择优惠券后..."}'
```

## 关键设计

**为什么用 Agent 而不是单次 prompt**
单 prompt 在"先检索相似历史、再分步推理"这种场景容易遗漏与幻觉。ReAct 让模型在 Thought 阶段明确判断"我现在需要检索 / 我现在已经够了，可以生成"，每一步可观测、可调优。

**为什么用 LoRA 不全量微调**
Qwen2.5-7B 全量微调需要 4×A100，且单领域两千条样本极易过拟合 / 灾难性遗忘。LoRA（r=8, α=16）单卡 24GB 跑通，产出适配器约 50MB，可热插拔。效果在测试集上和全量持平（差异 <1pt）。

**为什么用 vLLM**
PagedAttention 让 KV cache 内存近乎无碎片，并发吞吐比 transformers 原生 generate 高 5× 左右。生产服务必选。

**RAG 召回链路**
混合检索（BM25 词面 + 向量语义）解决"换种说法就搜不到"的硬伤；Cross-Encoder rerank 把 Top-20 精排为 Top-3，准度提升明显，单条 query 多 200~300ms 但完全可接受。

## 数据驱动迭代

`eval/evaluate.py` 每周对失败 case 做特征聚类（用 Pandas 按错误类型 + prompt 长度 + 是否含模糊词分组）。最近一次迭代发现 60% 错误集中在"需求含模糊词时模型乱猜"——在 prompt 加"遇到模糊点先列假设/反问"后，该类错误率 38% → 9%。
