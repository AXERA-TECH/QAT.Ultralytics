from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.load("yolo11n.pt")  # build from YAML and transfer weights

# Best 10-epoch QAT debug group (fixed to GPU 3).
results = model.train(
    data="coco.yaml",
    batch=64,
    epochs=10,
    imgsz=640,
    qat_onnx_imgsz=[640, 640],
    device=0,
    project="runs/detect",      # 保存目录
    name="qat",
    save_period=1,
    qat_onnx_sp="runs/last_checkpoint.onnx", # 训练完成后，导出的onnx
    lr0=0.00004,
    lrf=0.2,
    # fraction=1,  # 可设0.01训练数据，进行流程测试
)
