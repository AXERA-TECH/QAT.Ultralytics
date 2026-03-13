from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")

# Run batched inference on a list of images
results = model(["bus.jpg"],
                qat_onnx_imgsz=[640,640],
                qat_pt_path='./runs/detect/qat/weights/last.pt',
                )

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk