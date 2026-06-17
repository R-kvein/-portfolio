"""节点 ①：语义解析 —— 抽意图 + 关键实体。

用 LLM Function Call 抽实体比正则稳，
跨业务场景换词典只改 prompt，不改代码。
"""
from __future__ import annotations
import json
import re

from ..state import AnalysisState


def parse_node(state: AnalysisState) -> AnalysisState:
    q = state["question"]
    # 简化：实体抽取用规则 + LLM 兜底（这里给规则 baseline）
    entities = {}
    if m := re.search(r"订单[号]?\s*(\w+)", q):
        entities["order_id"] = m.group(1)
    if m := re.search(r"用户[ID|id]*\s*(\w+)", q):
        entities["user_id"] = m.group(1)
    if m := re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", q):
        entities["date"] = m.group(1)

    # 意图分类（实际可外挂一个小分类器，这里给关键词版）
    intent = "diagnose"
    if any(k in q for k in ["怎么办", "如何处理", "建议"]):
        intent = "suggest"
    elif any(k in q for k in ["对比", "区别"]):
        intent = "compare"

    return {"intent": intent, "entities": entities}
