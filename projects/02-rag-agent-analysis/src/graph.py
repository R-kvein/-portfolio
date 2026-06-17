"""LangGraph 状态机：分析引擎主流程。

边的设计原则：
- 正常路径线性向下，异常路径用条件边显式跳走，不在节点里写 if；
- retry 也是一条边（reason→reason），上限 2 次，超出走降级。

每个节点都是 pure function（输入 state，输出 partial state），
便于单测和观察 LangGraph trace。
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from .state import AnalysisState
from .nodes.parse import parse_node
from .nodes.retrieve import retrieve_node, cache_check_node
from .nodes.assemble import assemble_node, ask_user_node
from .nodes.reason import reason_node, format_node, degrade_node


# ---- 条件路由 ----
def route_after_cache(state: AnalysisState) -> str:
    return "format" if state.get("cached_answer") else "retrieve"


def route_after_retrieve(state: AnalysisState) -> str:
    return "assemble" if state.get("docs") else "ask_user"


def route_after_reason(state: AnalysisState) -> str:
    if state.get("error"):
        if state.get("retry_count", 0) < 2:
            return "reason"                       # retry
        return "degrade"                          # 降级
    return "format"


def build_graph():
    g = StateGraph(AnalysisState)

    g.add_node("parse",     parse_node)
    g.add_node("cache",     cache_check_node)
    g.add_node("retrieve",  retrieve_node)
    g.add_node("assemble",  assemble_node)
    g.add_node("ask_user",  ask_user_node)
    g.add_node("reason",    reason_node)
    g.add_node("degrade",   degrade_node)
    g.add_node("format",    format_node)

    g.add_edge(START, "parse")
    g.add_edge("parse", "cache")
    g.add_conditional_edges("cache",     route_after_cache,    {"retrieve": "retrieve", "format": "format"})
    g.add_conditional_edges("retrieve",  route_after_retrieve, {"assemble": "assemble", "ask_user": "ask_user"})
    g.add_edge("assemble", "reason")
    g.add_conditional_edges("reason",    route_after_reason,   {"reason": "reason", "degrade": "degrade", "format": "format"})
    g.add_edge("degrade",  "format")
    g.add_edge("ask_user", END)
    g.add_edge("format",   END)

    return g.compile()
