"""状态对象：在 LangGraph 节点之间流转的数据。

为什么用 TypedDict 而不是 BaseModel：
- LangGraph 对 TypedDict 原生支持，update 操作零开销；
- BaseModel 每次节点返回都会触发深拷贝校验，高 QPS 时是瓶颈。
"""
from __future__ import annotations
from typing import TypedDict, Optional


class AnalysisState(TypedDict, total=False):
    # 输入
    question: str

    # parse 阶段产出
    intent: str
    entities: dict

    # cache 命中（如有）
    cached_answer: Optional[dict]

    # retrieve 阶段产出
    docs: list[dict]               # [{id, content, score}]
    cited_ids: list[str]

    # assemble 阶段产出
    context: str

    # reason 阶段产出
    answer: dict                   # {reason, evidence, suggestions}
    retry_count: int

    # 错误兜底
    error: Optional[str]
    degraded: bool                 # 是否走了降级分支
