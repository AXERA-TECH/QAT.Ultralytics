from ultralytics import YOLO
# Load a model
model = YOLO("yolo11s.yaml")
model.load("yolo11s.pt")  # build from YAML and transfer weights
# Train the model
results = model.train(data="coco.yaml", batch=64, epochs=6, imgsz=640) #, device=[0, 1]