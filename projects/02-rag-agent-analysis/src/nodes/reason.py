"""节点 ⑤：LLM 推理 + 建议生成。

输出强制三段式：原因 / 证据 / 建议步骤。
- 不三段式 → parse 失败 → 节点返回 error → 路由到 retry 或 degrade。
"""
from __future__ import annotations
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..state import AnalysisState


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一位资深分析师。基于【上下文】回答用户问题，严格按 JSON 输出：\n"
     "{{\"reason\": \"...\", \"evidence\": [\"...\"], \"suggestions\": [\"...\"]}}\n"
     "禁止编造上下文外的内容；evidence 必须从上下文摘取原文片段。"),
    ("user", "【问题】{question}\n\n【上下文】\n{context}"),
])

_llm = ChatOpenAI(model="qwen-plus", temperature=0.2, timeout=30, max_retries=0)


def reason_node(state: AnalysisState) -> AnalysisState:
    retry = state.get("retry_count", 0) + 1
    try:
        resp = _llm.invoke(_PROMPT.format_messages(
            question=state["question"], context=state.get("context", ""),
        ))
        ans = json.loads(resp.content)
        # schema 校验
        for k in ("reason", "evidence", "suggestions"):
            if k not in ans:
                raise ValueError(f"missing key: {k}")
        return {"answer": ans, "retry_count": retry, "error": None}
    except Exception as e:
        return {"retry_count": retry, "error": str(e)}


def degrade_node(state: AnalysisState) -> AnalysisState:
    """LLM 重试都失败 → 降级：直接把检索原文按"证据"返回，至少不让流程崩。"""
    docs = state.get("docs", [])
    return {
        "answer": {
            "reason": "LLM 推理服务异常，已降级返回检索原文供参考。",
            "evidence": [d["content"] for d in docs[:3]],
            "suggestions": ["请人工查看原文后判断"],
        },
        "degraded": True,
    }


def format_node(state: AnalysisState) -> AnalysisState:
    """最终格式化：把 cited_ids 挂进 answer，前端可点击回溯原文。"""
    ans = state.get("cached_answer") or state.get("answer") or {}
    ans = dict(ans)
    ans["cited_ids"] = state.get("cited_ids", [])
    ans["degraded"]  = state.get("degraded", False)
    return {"answer": ans}
