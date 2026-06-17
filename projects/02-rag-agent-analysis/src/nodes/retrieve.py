"""节点 ②③：缓存检查 + 混合检索。

缓存命中是省 LLM 钱的最重要一环：
按 (intent, entities) 哈希做 key，命中率 ~18%，省下 1/5 的 LLM 调用。
"""
from __future__ import annotations
import hashlib
import json

from ..state import AnalysisState
from ..tools import term_normalizer, case_cache, hybrid_retrieve


def cache_check_node(state: AnalysisState) -> AnalysisState:
    key = _cache_key(state)
    hit = case_cache.get(key)
    return {"cached_answer": hit} if hit else {}


def retrieve_node(state: AnalysisState) -> AnalysisState:
    # 用术语归一化把"同义不同名"先抹平，再去检索
    q = term_normalizer.normalize(state["question"])
    docs = hybrid_retrieve(q, top_k=5)
    return {"docs": docs, "cited_ids": [d["id"] for d in docs]}


def _cache_key(state: AnalysisState) -> str:
    payload = json.dumps(
        {"intent": state.get("intent"), "entities": state.get("entities", {})},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
