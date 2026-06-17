"""FastAPI 包装。LangGraph 编译出的 app 直接 invoke 即可。"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .graph import build_graph


class Req(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


app = FastAPI(title="RAG + Agent Analysis Engine")
graph = build_graph()


@app.post("/analyze")
def analyze(req: Req):
    out = graph.invoke({"question": req.question, "retry_count": 0})
    return {"answer": out.get("answer"), "intent": out.get("intent"),
            "degraded": out.get("degraded", False)}


@app.get("/health")
def health(): return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8002)
