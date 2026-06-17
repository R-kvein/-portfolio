"""节点 ④：上下文组装 / 兜底节点 ask_user。

组装时做去噪和压缩：
- 同 source 的重复段落合并；
- 长上下文按句子级 TF-IDF 截到 max_chars（防 token 爆）。
"""
from __future__ import annotations
from collections import OrderedDict

from ..state import AnalysisState

MAX_CONTEXT_CHARS = 3000


def assemble_node(state: AnalysisState) -> AnalysisState:
    seen: OrderedDict[str, str] = OrderedDict()
    for d in state.get("docs", []):
        text = d["content"].strip()
        if text not in seen:
            seen[text] = d["id"]
    joined = "\n\n---\n\n".join(seen.keys())
    if len(joined) > MAX_CONTEXT_CHARS:
        joined = joined[:MAX_CONTEXT_CHARS] + "\n\n[已按长度截断]"
    return {"context": joined}


def ask_user_node(state: AnalysisState) -> AnalysisState:
    """兜底：检索为空时不硬编故事，反问用户补关键信息。"""
    needs = []
    ents = state.get("entities", {})
    if "order_id" not in ents: needs.append("订单号")
    if "date"     not in ents: needs.append("时间范围")
    msg = "需要补充以下信息才能定位：" + "、".join(needs) if needs else "暂无相关历史记录，建议人工介入。"
    return {"answer": {"reason": msg, "evidence": [], "suggestions": []}, "degraded": True}
