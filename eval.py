from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")
model.load("yolo11n.pt")  # build from YAML and transfer weights

# Validate the model
metrics = model.val(data="coco.yaml",   # 数据集
                    batch=16,   #
                    qat_pt_path='runs/qat_test/best_qat/weights/best.pt',   # 评估qat模型路径
                    qat_onnx_imgsz=[640, 640],                              # Onnx模型的shape, [h, w]
                    device=2,   # 评估设备
                    )  # no arguments needed, dataset and settings remembered 
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category

# yolov11官方精度
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.466
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.635
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.503
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.292
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.511
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.362
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.598
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.651
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.472
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.709
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.813