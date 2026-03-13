from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.load("yolo11n.pt")  # build from YAML and transfer weights

# Best 10-epoch QAT debug group (fixed to GPU 3).
results = model.train(
    data="coco.yaml",
    batch=128,
    epochs=12,
    imgsz=640,
    qat_onnx_imgsz=[640, 640],
    device=[2,3],
    project="runs/detect",      # 保存目录
    name="qat",
    save_period=1,
    qat_onnx_sp="runs/last_checkpoint.onnx", # 训练完成后，导出的onnx
    lr0=0.00003,
    lrf=0.1,
    # fraction=0.01,  # 1%训练数据，进行流程测试
)

# python -m torch.distributed.run --nproc_per_node 2 train-gpus.py
