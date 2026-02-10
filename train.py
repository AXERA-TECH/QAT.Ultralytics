from ultralytics import YOLO
# Load a model
model = YOLO("yolo11s.yaml")
model.load("yolo11s.pt")  # build from YAML and transfer weights
# Train the model
results = model.train(data="coco.yaml", # 数据集
                      batch=64,     # 
                      epochs=30,     # 据实际情况适当增加 
                      imgsz=640,    # 训练最大边长
                      qat_onnx_imgsz=[640, 640],  # onnx导出shape,格式为[h, w]
                      qat_onnx_sp='./last_checkpoint.onnx', # 导出onnx保存路径
                      device=[0, 1]
                      ) #, device=[0, 1]