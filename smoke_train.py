"""端到端冒烟:真实 YOLO11 QAT 训练 1 epoch(coco 极小 fraction),
验证迁移后的 torch2.10 QAT 全链路(capture→prepare_qat_pt2e→训练→convert→导出)。
run: cd <repo> && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python smoke_train.py
"""
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.load("yolo11n.pt")

results = model.train(
    data="coco.yaml",
    batch=4,
    epochs=1,
    imgsz=640,
    qat_onnx_imgsz=[640, 640],
    device=0,                 # 配合 CUDA_VISIBLE_DEVICES=2 → 物理 GPU2
    project="runs/detect",
    name="qat_smoke",
    exist_ok=True,
    fraction=0.003,           # ~350 张,打通链路用
    workers=2,
    qat_onnx_sp="runs/qat_smoke.onnx",
    lr0=0.00004,
    lrf=0.2,
    save_period=1,
)
print("=== TRAIN DONE ===")
