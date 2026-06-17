"""FastAPI 服务入口。

接口设计：
- POST /usecase: 需求 → 结构化用例（同步）
- GET  /health: 健康检查（K8s/Docker 拉起后用）

为什么是 FastAPI：
- 异步原生支持，未来要接流式输出（SSE）方便升级；
- Pydantic 自动校验入参，少写 100 行手动校验。
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import build_agent, run_one


# 启动时把语料库加载到内存里建索引，免每次请求重建
_CORPUS_FILE = Path(__file__).parent.parent / "data" / "history_corpus.txt"


class Req(BaseModel):
    requirement: str = Field(..., min_length=1, max_length=5000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    corpus = _load_corpus()
    app.state.agent = build_agent(corpus)
    yield


app = FastAPI(title="Agent + RAG Usecase API", lifespan=lifespan)


@app.post("/usecase")
def usecase(req: Req):
    return run_one(app.state.agent, req.requirement)


@app.get("/health")
def health():
    return {"status": "ok"}


def _load_corpus() -> list[str]:
    if not _CORPUS_FILE.exists():
        # 冷启动 demo 数据，生产从 OSS/PostgreSQL 加载
        return [
            "结算页优惠券核销规则：用户可叠加 1 张满减券与 1 张品类券，..." ,
            "支付失败重试策略：首次失败 5 秒重试，第二次 30 秒，..." ,
            "购物车合并机制：登录后游客购物车与账号购物车按 SKU 去重合并，..." ,
        ]
    return [l.strip() for l in _CORPUS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)
