"""集中配置：路径、模型、检索阈值。所有调参点在一个文件里，便于实验追踪。"""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- LLM (vLLM OpenAI-compatible endpoint) ----
LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")
LLM_API_KEY   = os.getenv("LLM_API_KEY", "EMPTY")          # vLLM 无鉴权时占位
LLM_MODEL     = os.getenv("LLM_MODEL", "usecase")          # 与 vllm serve --lora-modules 一致
LLM_TEMPERATURE = 0.2                                      # 防幻觉：低温

# ---- Embedding / Rerank ----
EMBED_MODEL   = "BAAI/bge-large-zh-v1.5"
RERANK_MODEL  = "BAAI/bge-reranker-large"

# ---- 检索 ----
CHROMA_DIR    = str(BASE_DIR / "data" / "chroma")
CHUNK_SIZE    = 600          # 调参实验显示 600 + overlap 100 最优
CHUNK_OVERLAP = 100
TOPK_RECALL   = 20           # 粗召回数量
TOPK_RERANK   = 3            # 精排后给 LLM 的数量
BM25_WEIGHT   = 0.4          # 混合检索权重
VEC_WEIGHT    = 0.6

# ---- Agent ----
MAX_ITERATIONS = 6           # ReAct 最大循环步数，防卡死
