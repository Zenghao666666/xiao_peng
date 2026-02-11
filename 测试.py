from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("C:/Users/13380/Desktop/text1/54.bmp", save=True, imgsz=320, conf=0.5,)