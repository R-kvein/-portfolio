"""ReAct Agent 主循环。

为什么用 ReAct 而不是单次 prompt：
- 需求 → 用例 需要"先检索相似历史 + 分步推理"，单 prompt 易遗漏/幻觉；
- ReAct 让模型在 Thought 阶段明确决定调哪个工具，每一步可观测、可调优；
- 工具描述（tools.py 里的 description）= Agent 的"判断依据"，写得好不好直接决定选对率。

为什么限制 MAX_ITERATIONS：
- 防止 Agent 在某个 Thought 上反复横跳卡死；
- 经验值 6 足够应付绝大部分需求（实测 P95 = 4 步）。
"""
from __future__ import annotations
from typing import Iterable

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from .llm import get_llm
from .retriever import HybridRetriever
from .tools import build_all_tools
from . import config


REACT_PROMPT = PromptTemplate.from_template(
    """你是资深业务分析师。回答用户的需求转换任务，可调用以下工具：

{tools}

严格按如下格式输出：

Question: 用户需求
Thought: 我现在要做什么？要不要调工具？调哪个？
Action: 工具名（必须是 [{tool_names}] 中之一）
Action Input: 工具入参
Observation: 工具返回
... (Thought/Action/Action Input/Observation 可重复)
Thought: 我已经有足够信息生成最终用例。
Final Answer: 最终结构化用例 JSON

约束：
- 必须先 history_search 检索历史，再 usecase_generate 生成；
- 生成后建议 mcp_internal 做合规校验，如失败回退重新生成；
- 遇到模糊需求，在用例 steps 里以 [假设] 显式列出，不要编造。

Question: {input}
{agent_scratchpad}"""
)


def build_agent(corpus: Iterable[str], metas: Iterable[dict] | None = None) -> AgentExecutor:
    """组装可执行的 Agent。"""
    retriever = HybridRetriever(list(corpus), list(metas) if metas else None)
    tools = build_all_tools(retriever)
    llm = get_llm()
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,                              # 生产关掉，开发开
        max_iterations=config.MAX_ITERATIONS,
        handle_parsing_errors=True,                 # LLM 偶尔格式跑偏，自动纠错
        return_intermediate_steps=True,             # 暴露中间 Thought/Action 给前端展示
    )


def run_one(executor: AgentExecutor, requirement: str) -> dict:
    """执行一次需求转换，返回最终用例 + 中间步骤（便于审计）。"""
    result = executor.invoke({"input": requirement})
    return {
        "usecase": result["output"],
        "steps": [
            {"action": s[0].tool, "input": s[0].tool_input, "observation": s[1]}
            for s in result.get("intermediate_steps", [])
        ],
    }
