# YOLOv8 图像识别与可视化平台

数据标注 → 训练 → 推理服务 → Web 可视化的完整目标检测项目。

## 成果

| 指标 | 值 |
|---|---|
| mAP@0.5 | **0.83** |
| mAP@0.5:0.95 | 0.61 |
| 推理延迟（CPU 单图） | ~120 ms |
| 推理延迟（GPU 单图） | ~18 ms |

## 全流程

```
labelImg 标注 → YOLO 格式数据集 → ultralytics 训练 → 调超参看 mAP
                                                    │
                                                    ▼
                                           best.pt 模型
                                                    │
                                                    ▼
                                 FastAPI 推理服务 + 前端上传可视化
```

## 关键调参经验

- **学习率**：0.01 → 0.005 后 mAP 提升 3pt（默认 0.01 在我们数据集上震荡）
- **图像尺寸**：416 → 640，mAP +5pt，但训练慢 2×（业务接受度可）
- **类别不平衡**：小类样本 ×3 增广（mosaic + mixup），mAP +2pt
- **epoch**：50 → 100 收益递减，过拟合风险增加；用 patience=20 早停

## 目录结构

```
.
├── src/
│   ├── train.py      # 训练脚本
│   ├── infer.py      # 单图/批量推理
│   ├── viz.py        # bbox 可视化
│   └── server.py     # FastAPI: 上传图片 → 返回标注后的图 + JSON
├── data/
│   └── data.yaml     # YOLOv8 数据集配置
└── requirements.txt
```

## 跑起来

```bash
pip install -r requirements.txt

# 训练
python src/train.py --data data/data.yaml --epochs 100 --imgsz 640

# 服务
python src/server.py
# 浏览器访问 http://localhost:8003/docs 试一下
```
