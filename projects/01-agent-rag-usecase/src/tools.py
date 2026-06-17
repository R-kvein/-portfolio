"""三类 Tool 注册给 ReAct Agent。

工具描述决定 Agent 选对率——这里每个 description 都明确写出
"适用场景 / 不适用场景 / 入参格式"，这是把准确率从 60% 推到 88% 的关键之一。
"""
from __future__ import annotations
import json
from typing import List

from langchain.tools import Tool
from pydantic import BaseModel

from .llm import get_llm
from .retriever import HybridRetriever, Doc


# ---------- ① RAG 检索工具 ----------
def make_rag_tool(retriever: HybridRetriever) -> Tool:
    def _run(query: str) -> str:
        docs: List[Doc] = retriever.search(query)
        if not docs:
            return "NO_HISTORY_FOUND"
        # 把检索结果序列化回 Agent，附 score 让 LLM 自己判断置信度
        return "\n---\n".join(
            f"[score={d.score:.2f}] {d.content}" for d in docs
        )

    return Tool(
        name="history_search",
        func=_run,
        description=(
            "检索与当前需求相似的历史用例和已有模块文档。"
            "适用：用户需求涉及已有模块、术语、流程时优先调用。"
            "不适用：纯算法/通用知识类问题，不必查史。"
            "入参：完整的需求描述字符串。"
        ),
    )


# ---------- ② 结构化用例生成工具 ----------
class UseCase(BaseModel):
    title: str
    actor: str
    preconditions: list[str]
    steps: list[str]
    expected_result: str


def make_usecase_tool() -> Tool:
    llm = get_llm(temperature=0.2)
    system = (
        "你是资深业务分析师。基于给定的需求和检索到的历史用例上下文，"
        "输出严格符合 JSON Schema 的结构化用例（UseCase）。"
        "遇到模糊点先列假设：在 steps 里以 '[假设] ...' 开头。"
        "禁止编造历史上下文里没有的术语。"
    )

    def _run(payload: str) -> str:
        # payload = JSON: {"requirement": "...", "context": "..."}
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {"requirement": payload, "context": ""}
        user = (
            f"【需求】{data.get('requirement','')}\n\n"
            f"【历史上下文】\n{data.get('context','(无)')}\n\n"
            "请输出 UseCase JSON。"
        )
        resp = llm.invoke([("system", system), ("user", user)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        # 校验：不合规则触发 Agent 重试
        try:
            UseCase.model_validate_json(text)
        except Exception as e:
            return f"INVALID_OUTPUT: {e}\nRAW:{text}"
        return text

    return Tool(
        name="usecase_generate",
        func=_run,
        description=(
            "基于已检索的历史上下文，生成最终结构化用例 JSON。"
            "适用：必须先调 history_search 拿到上下文后再调用本工具。"
            "不适用：上下文不足或冲突时，先回 history_search 补检索。"
            "入参：JSON 字符串 {requirement: str, context: str}"
        ),
    )


# ---------- ③ MCP 内部工具（合规校验） ----------
def make_mcp_tool() -> Tool:
    """通过 MCP 协议连接内部术语库 / 合规规则引擎。
    这里为演示给出本地 stub；生产环境替换为 MCP client 调用。"""
    BANNED_TERMS = {"内部代号XX", "未公开项目Y"}

    def _run(text: str) -> str:
        hit = [t for t in BANNED_TERMS if t in text]
        if hit:
            return f"COMPLIANCE_FAIL: 含禁用词 {hit}"
        return "COMPLIANCE_OK"

    return Tool(
        name="mcp_internal",
        func=_run,
        description=(
            "调用内部 MCP Server（术语库 + 合规规则）。"
            "适用：当生成的用例文本需要做术语合规校验时调用。"
            "不适用：仅检索和生成阶段，不必每步都调。"
            "入参：完整待校验文本。"
        ),
    )


def build_all_tools(retriever: HybridRetriever) -> list[Tool]:
    return [make_rag_tool(retriever), make_usecase_tool(), make_mcp_tool()]
