import cv2
import os
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO('2.pt')  # 这里替换成你的 YOLOv8 模型路径

# 定义图片文件夹路径和保存路径
image_folder2 = 'D:/6.15/最终检测/picture'  # 替换成你的图片文件夹路径
save_folder2 = 'D:/6.15/最终检测/dic'  # 替换成你想要保存检测结果的文件夹路径

# 创建保存文件夹（如果不存在）
os.makedirs(save_folder2, exist_ok=True)

# 遍历图片文件夹中的所有图片
for filename in os.listdir(image_folder2):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 支持常见的图片格式
        # 构建图片完整路径
        image_path = os.path.join(image_folder2, filename)

        # 读取图片
        image = cv2.imread(image_path)

        # 进行目标检测
        results = model(image)

        # 获取检测结果（这里 results 是一个列表，每个元素对应一张图片的结果）
        detections = results[0].boxes  # 获取第一个图片的检测框信息

        # 在图片上绘制检测框和标签
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
            label = f"{model.names[box.cls[0].item()]} {box.conf[0].item():.2f}"  # 获取标签和置信度
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 保存检测结果
        save_path = os.path.join(save_folder2, filename)
        cv2.imwrite(save_path, image)

        print(f"Processed and saved image {filename}")