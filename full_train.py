"""全量 COCO QAT 训练,复现 README yolo11n 精度(目标 8w8f: mAP50-95≈0.386 / mAP50≈0.546)。
配置同 README/train.py:batch64 / 10 epoch / imgsz640 / lr0 4e-5 / lrf 0.2。
run: cd <repo> && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. nohup python -u full_train.py > /tmp/qax_full.log 2>&1 &
"""
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.load("yolo11n.pt")

results = model.train(
    data="coco.yaml",          # 全量 COCO(118k train / 5k val)
    batch=64,
    epochs=10,
    imgsz=640,
    qat_onnx_imgsz=[640, 640],
    device=0,                  # 配合 CUDA_VISIBLE_DEVICES=3 → 物理 GPU3
    project="runs/detect",
    name="qat_full",
    exist_ok=True,
    workers=8,
    save_period=-1,            # 只存 best/last,省 NFS 配额
    qat_onnx_sp="runs/qat_full.onnx",
    lr0=0.00004,
    lrf=0.2,
)
print("=== FULL TRAIN DONE ===")
print(results)
