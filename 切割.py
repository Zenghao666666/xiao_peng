from ultralytics import YOLO
import cv2
import os
import numpy as np

# 初始化YOLOv8模型，这里使用yolov8n.pt作为示例
model = YOLO("1.pt")

# 指定包含图片的文件夹路径
image_folder = 'D:/6.15/待测试文件夹'

# 指定裁剪后图片的保存目录
crops_folder = 'D:/6.15/最终检测/picture'
if not os.path.exists(crops_folder):
    os.makedirs(crops_folder)

# 获取所有图片文件的路径
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
               f.endswith(('jpg', 'png', 'jpeg', 'bmp'))]

# 遍历所有图片文件
for image_file in image_files:
    # 读取图片
    im0 = cv2.imread(image_file)
    if im0 is None:
        print(f"Warning: {image_file} is not a valid image and will be skipped.")
        continue

    # 使用模型进行预测，不显示结果
    results = model.predict(source=image_file, show=False)  # 确保source参数正确传递

    # 检查results是否为列表
    if isinstance(results, list):
        # 遍历每个检测结果
        for result in results:
            # 获取边界框和类别
            boxes = result.boxes.xyxy.cpu().numpy()  # 获取xyxy格式的边界框坐标
            clss = result.boxes.cls.cpu().numpy()    # 获取检测到的类别编号

            # 裁剪和保存图片
            for idx, (box, cls) in enumerate(zip(boxes, clss)):
                x1, y1, x2, y2 = box
                crop_img = im0[int(y1):int(y2), int(x1):int(x2)]

                # 保存裁剪后的图片
                save_path = os.path.join(crops_folder, f"{os.path.basename(image_file).split('.')[0]}_{idx}_{int(cls)}.png")
                cv2.imwrite(save_path, crop_img)
                print(f"Saved cropped image to {save_path}")
    else:
        print(f"No detection results for {image_file}")

print("Image processing completed.")


