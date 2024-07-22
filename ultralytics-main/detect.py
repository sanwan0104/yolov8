import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 加载模型
model = YOLO("D:\\machine_learning\\yolo8\\ultralytics-main\\runs\\detect\\train3\\weights\\best.pt")

# 进行预测
results = model.predict(source="datasets/test/images", imgsz=320, show=False, save=True)

# 提取预测置信度
confidences = []
for result in results:
    for pred in result.boxes:
        # 将张量从CUDA设备转移到CPU并转换为NumPy数组
        confidences.append(pred.conf.cpu().numpy())

# 绘制预测置信曲线
plt.figure(figsize=(10, 6))
plt.plot(confidences, marker='o')
plt.title('Prediction Confidence Curve')
plt.xlabel('Prediction Index')
plt.ylabel('Confidence')
plt.grid(True)
plt.show()