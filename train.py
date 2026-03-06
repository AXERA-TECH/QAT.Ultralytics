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
    qat_onnx_sp="./last_checkpoint.onnx",
    device=[3],
    project="runs/detect",
    name="best_qat_e10",
    lr0=0.00002,
    lrf=0.1,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    close_mosaic=0,
    qat_enable_fake_quant_epoch=0,
    qat_disable_observer_epoch=-1,
    qat_disable_fake_quant_epoch=-1,
)
