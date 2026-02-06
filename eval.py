from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.yaml")
# model = YOLO("yolo11s.pt")  # load an official model

# Validate the model
metrics = model.val(data="coco.yaml",   # 数据集
                    batch=16,   #
                    qat_pt_path='./last_checkpoint.pth',    # 评估qat模型路径
                    qat_onnx_sp='./last_checkpoint.onnx',   # 导出Onnx模型，并导出{last_checkpoint}_qat_slim.onnx
                    qat_onnx_imgsz=[640, 640],              # 导出Onnx模型的shape, [h, w]
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