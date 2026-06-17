"""在原图上画检测框 → 返回 PIL Image / bytes，前端直接展示。"""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


_COLORS = ["#7cc4ff", "#9eff9a", "#ffb86b", "#ff8baf", "#c891ff"]


def draw_detections(image_path: str | Path, detections: list[dict]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, det in enumerate(detections):
        color = _COLORS[det["cls"] % len(_COLORS)]
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['name']} {det['conf']:.2f}"
        draw.rectangle([x1, y1 - 18, x1 + 8 * len(label), y1], fill=color)
        draw.text((x1 + 2, y1 - 16), label, fill="black")
    return img
