from ultralytics import YOLO

# Best-known 10-epoch QAT baseline on ultralytics/cfg/datasets/coco.yaml
model = YOLO('yolo11n.yaml')
model.load('yolo11n.pt')

model.train(
    data='ultralytics/cfg/datasets/coco.yaml',
    batch=64,
    epochs=10,
    imgsz=640,
    device=[3],
    optimizer='SGD',
    lr0=4e-5,
    lrf=0.2,
    cos_lr=False,
    warmup_epochs=0.0,
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
    qat_enable_fake_quant_epoch=7,
    qat_disable_observer_epoch=9,
    qat_disable_fake_quant_epoch=-1,
    qat_onnx_imgsz=[640, 640],
    qat_onnx_sp='./runs/qat_test/best_qat/exp_train2017_best_03864.onnx',
    project='runs/qat_test',
    name='best_qat',
    exist_ok=False,
    save_period=1,
)
