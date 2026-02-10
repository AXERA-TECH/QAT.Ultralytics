from ultralytics import YOLO
model = YOLO("yolo11s.yaml")
model.export(batch=1,   #
            format='axera',
            simplify=True,
            device='cuda',
            qat_pt_path='./last_checkpoint.pth',    # 评估qat模型路径
            qat_onnx_sp='./last_checkpoint.onnx',   # 导出Onnx模型，并导出{last_checkpoint}_qat_slim.onnx
            qat_onnx_imgsz=[640, 640],              # 导出Onnx模型的shape, [h, w]
            )  # no arguments needed, dataset and settings remembered 