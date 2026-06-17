"""5 大业务痛点对应的工具集。

工具拆分原则：每个工具只解决一件事，便于在不同节点里组合调用。
"""
from __future__ import annotations
from dataclasses import dataclass


# ============ 痛点 1：术语归一化 ============
class TermNormalizer:
    """业务词典命中 → 直接替换；未命中 → fallback 到 Embedding 相似度回退。"""
    _DICT = {
        "下单失败": "order_create_failed",
        "支付未到账": "payment_pending",
        "退款卡住": "refund_stuck",
        "卡顿":     "latency_high",
    }

    def normalize(self, text: str) -> str:
        for raw, std in self._DICT.items():
            text = text.replace(raw, std)
        return text

term_normalizer = TermNormalizer()


# ============ 痛点 2：日志结构化抽取 ============
def log_extractor(raw_log: str) -> dict:
    """从异构日志里抽出统一 schema。生产里用 LLM Function Call。"""
    import re
    return {
        "ts":      (re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", raw_log) or [""])[0],
        "level":   (re.search(r"\b(ERROR|WARN|INFO)\b", raw_log) or [""])[0],
        "service": (re.search(r"\[([\w-]+)\]", raw_log) or [None, ""])[1],
        "message": raw_log.strip()[-200:],
    }


# ============ 痛点 3：高频 case 缓存 ============
class CaseCache:
    """内存版本；生产用 Redis + TTL。"""
    def __init__(self): self._store: dict = {}
    def get(self, key: str): return self._store.get(key)
    def set(self, key: str, val: dict): self._store[key] = val

case_cache = CaseCache()


# ============ 痛点 4：输出约束（三段式） ============
def output_constrainer(text: str) -> dict | None:
    """把 LLM 自由文本强制拆成 reason/evidence/suggestions。"""
    import re
    rs = re.search(r"原因[：:](.*?)(?=证据|建议|$)", text, re.S)
    ev = re.search(r"证据[：:](.*?)(?=建议|$)", text, re.S)
    sg = re.search(r"建议[：:](.*)$", text, re.S)
    if not (rs and sg):
        return None
    return {
        "reason":      rs.group(1).strip(),
        "evidence":    [e.strip() for e in (ev.group(1).split("\n") if ev else []) if e.strip()],
        "suggestions": [s.strip() for s in sg.group(1).split("\n") if s.strip()],
    }


# ============ 痛点 5：citation 标注 ============
def citation_tagger(answer: dict, doc_ids: list[str]) -> dict:
    answer = dict(answer)
    answer["citations"] = doc_ids
    return answer


# ============ 混合检索（简化 stub，实际复用 Project 1 的 HybridRetriever）============
@dataclass
class _Doc:
    id: str; content: str; score: float

def hybrid_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """演示用 stub。生产复用 projects/01-agent-rag-usecase/src/retriever.py 的 HybridRetriever。"""
    fake_docs = [
        _Doc("hist_001", f"历史 case：{query[:30]} 通常由 service 超时引起", 0.92),
        _Doc("hist_002", f"日志样例：[order-service] ERROR 调用 payment 超时 30s",  0.88),
        _Doc("hist_003", f"经验记录：重启 payment-gw 后恢复",                       0.75),
    ]
    return [{"id": d.id, "content": d.content, "score": d.score} for d in fake_docs[:top_k]]
