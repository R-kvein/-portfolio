"""单图 / 批量推理。被 server.py 调用。"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


_MODEL_CACHE: dict[str, YOLO] = {}


def get_model(weights: str = "runs/detect/train/weights/best.pt") -> YOLO:
    """模型懒加载 + 进程内缓存，避免每次请求重载。"""
    if weights not in _MODEL_CACHE:
        if not Path(weights).exists():
            weights = "yolov8n.pt"            # demo fallback：用预训练
        _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


def predict(image_path: str | Path, conf: float = 0.25) -> list[dict]:
    model = get_model()
    res = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    return [
        {
            "cls":   int(b.cls.item()),
            "name":  res.names[int(b.cls.item())],
            "conf":  round(float(b.conf.item()), 3),
            "bbox":  [round(float(x), 1) for x in b.xyxy[0].tolist()],
        }
        for b in res.boxes
    ]


def predict_batch(paths: Iterable[str | Path], conf: float = 0.25) -> list[list[dict]]:
    return [predict(p, conf=conf) for p in paths]
