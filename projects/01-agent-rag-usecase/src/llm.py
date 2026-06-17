"""LLM 客户端封装：用 OpenAI 兼容接口对接 vLLM，LoRA 适配器通过 model 名切换。

为什么这么写：
1. vLLM 暴露 OpenAI 兼容 API，langchain_openai 直接复用，无需为 vLLM 单独写适配；
2. LoRA 在 vLLM 端通过 --lora-modules 注册，调用方只改 model 参数即可热切换；
3. 低 temperature(0.2) + 强 system prompt 是防幻觉的第一道闸。
"""
from langchain_openai import ChatOpenAI
from . import config


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        timeout=60,
        max_retries=2,
    )
