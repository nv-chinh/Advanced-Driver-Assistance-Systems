from ultralytics import YOLO
# Load model
model = YOLO("yolov8l.pt")

# Export the model to ONNX format
model.export(format="onnx", imgsz = [480, 640], device = '0')