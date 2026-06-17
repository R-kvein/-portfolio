"""YOLOv8 训练入口。

调参经验值都在默认参数里写清了，
改一个跑一组实验，对比 runs/detect/*/results.csv 看 mAP 变化。
"""
from __future__ import annotations
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data/data.yaml")
    ap.add_argument("--model",  default="yolov8n.pt", help="n/s/m/l/x — 越大越准越慢")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz",  type=int, default=640, help="实验值 416→640 给 +5pt mAP")
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr0",    type=float, default=0.005, help="实验值：默认 0.01 在我们数据集上震荡")
    ap.add_argument("--patience", type=int, default=20, help="早停，防过拟合")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        mosaic=1.0,      # 类别不平衡时的关键增广
        mixup=0.15,
        verbose=True,
    )
    # 验证并打印 mAP
    metrics = model.val()
    print(f"mAP@0.5      = {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95 = {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
