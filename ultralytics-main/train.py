from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = YOLO("weights/yolov8n.pt")  # 用于迁移训练的权重文件路径

results = model.train(data="datasets/MyData.yaml", imgsz=320, epochs=100, batch=16, device=0, workers=0)