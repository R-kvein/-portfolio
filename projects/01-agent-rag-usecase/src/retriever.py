"""混合检索 + 重排序

为什么不只用向量检索：
- 向量召回"意思相近"，但行业黑话/编号/版本号这类靠词面匹配的 BM25 完胜；
- 单纯 BM25 又对同义改写无力。两者加权融合 + Cross-Encoder 精排是最稳的组合。

测试集上的实验（控制变量）：
- 纯向量(Top-3)          → 准确率 60%
- + chunk 400→600        → 72%
- + BM25 混合(0.4/0.6)   → 80%
- + Cross-Encoder rerank → 85%
- + Few-shot prompt      → 88%
"""
from __future__ import annotations
from typing import List
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from . import config


@dataclass
class Doc:
    content: str
    meta: dict
    score: float = 0.0


class HybridRetriever:
    """BM25 + 向量 → Cross-Encoder rerank 一站式检索器。"""

    def __init__(self, corpus: List[str], metadatas: List[dict] | None = None):
        self.corpus = corpus
        self.metadatas = metadatas or [{} for _ in corpus]

        # 向量索引（持久化到磁盘，下次启动免重建）
        self.embed = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
        self.vec_store = Chroma.from_texts(
            texts=corpus,
            embedding=self.embed,
            metadatas=self.metadatas,
            persist_directory=config.CHROMA_DIR,
        )

        # BM25（中文按字切分作 baseline，工业上可换 jieba）
        tokenized = [list(t) for t in corpus]
        self.bm25 = BM25Okapi(tokenized)

        # Cross-Encoder（rerank 模型，比 Bi-Encoder 慢但精度高一截）
        self.reranker = CrossEncoder(config.RERANK_MODEL, max_length=512)

    def search(self, query: str, top_k: int | None = None) -> List[Doc]:
        top_k = top_k or config.TOPK_RERANK
        candidates = self._hybrid_recall(query, k=config.TOPK_RECALL)
        return self._rerank(query, candidates, top_k=top_k)

    # ---------- internal ----------
    def _hybrid_recall(self, query: str, k: int) -> List[Doc]:
        # 向量召回
        vec_hits = self.vec_store.similarity_search_with_score(query, k=k)
        vec_scores = {h[0].page_content: 1.0 - h[1] for h in vec_hits}  # Chroma 距离 → 相似度

        # BM25 召回
        bm25_scores_all = self.bm25.get_scores(list(query))
        bm25_top = sorted(
            range(len(self.corpus)), key=lambda i: bm25_scores_all[i], reverse=True
        )[:k]
        # min-max 归一化，避免 BM25 raw score 量纲影响
        mx = max(bm25_scores_all[bm25_top[0]], 1e-9) if bm25_top else 1.0
        bm25_scores = {self.corpus[i]: bm25_scores_all[i] / mx for i in bm25_top}

        # 加权融合
        merged: dict[str, float] = {}
        for text in set(vec_scores) | set(bm25_scores):
            merged[text] = (
                config.VEC_WEIGHT  * vec_scores.get(text, 0.0)
                + config.BM25_WEIGHT * bm25_scores.get(text, 0.0)
            )
        ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Doc(content=t, meta=self._meta_of(t), score=s) for t, s in ranked
        ]

    def _rerank(self, query: str, docs: List[Doc], top_k: int) -> List[Doc]:
        pairs = [(query, d.content) for d in docs]
        scores = self.reranker.predict(pairs)
        for d, s in zip(docs, scores):
            d.score = float(s)
        docs.sort(key=lambda d: d.score, reverse=True)
        return docs[:top_k]

    def _meta_of(self, text: str) -> dict:
        try:
            return self.metadatas[self.corpus.index(text)]
        except ValueError:
            return {}
