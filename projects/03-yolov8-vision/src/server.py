"""FastAPI 推理服务：上传图片 → 返回 JSON 检测结果 + 可视化图。"""
from __future__ import annotations
import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse

from .infer import predict
from .viz import draw_detections


app = FastAPI(title="YOLOv8 Vision API")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """返回 JSON 检测结果。"""
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "img.jpg").suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    return JSONResponse({"detections": predict(path)})


@app.post("/detect_viz")
async def detect_viz(file: UploadFile = File(...)):
    """返回画好框的图片（PNG bytes 流）。"""
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "img.jpg").suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    dets = predict(path)
    img = draw_detections(path, dets)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/health")
def health(): return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8003)
