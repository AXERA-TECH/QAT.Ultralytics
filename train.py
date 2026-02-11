from ultralytics import YOLO
# Load a model
# model = YOLO("yolo11s.yaml")
# model.load("yolo11s.pt")  # build from YAML and transfer weights
# # Train the model
# results = model.train(data="coco.yaml", # 数据集
#                       batch=64,     # 
#                       epochs=30,     # 据实际情况适当增加 
#                       imgsz=640,    # 训练最大边长
#                       qat_onnx_imgsz=[640, 640],  # onnx导出shape,格式为[h, w]
#                       qat_onnx_sp='./last_checkpoint.onnx', # 导出onnx保存路径
#                       device=[0, 1]
#                       ) #, device=[0, 1]

model = YOLO("yolo11n.yaml")
model.load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model with optimized QAT hyperparameters (参考YOLOv5 QAT)
results = model.train(
    data="coco.yaml",
    batch=64,
    epochs=50,              # 与YOLOv5 QAT一致
    imgsz=640,
    qat_onnx_imgsz=[640, 640],
    qat_onnx_sp='./last_checkpoint.onnx',
    device=[1],
    # ===== 参考YOLOv5 QAT配置 =====
    lr0=0.00001,            # 与YOLOv5 QAT一致
    lrf=0.1,                # 与YOLOv5 QAT一致
    # 注意：不要使用cos_lr，与YOLOv5保持一致
    # ===== 数据增强全部关闭（与YOLOv5 hyp.no-augmentation.yaml一致） =====
    hsv_h=0.0,              # 无颜色增强
    hsv_s=0.0,              # 无饱和度增强
    hsv_v=0.0,              # 无亮度增强
    degrees=0.0,            # 无旋转
    translate=0.0,          # 无平移
    scale=0.0,              # 无尺度变化
    fliplr=0.0,             # 无翻转
    mosaic=0.0,             # 无马赛克增强
    mixup=0.0,              # 无混合增强
    close_mosaic=10,        # 最后10轮关闭mosaic
)