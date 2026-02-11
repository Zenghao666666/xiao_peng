import tkinter as tk
from tkinter import ttk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Image Processing")

        # 创建和布局主要的框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 按钮布局
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, sticky=tk.W + tk.E)

        # 开始按钮
        self.start_button = ttk.Button(self.button_frame, text="开始处理", command=self.start_processing)
        self.start_button.pack(pady=10)

        # 上一张/下一张按钮
        self.prev_button = ttk.Button(self.button_frame, text="上一张", command=self.show_previous_image,
                                      state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(self.button_frame, text="下一张", command=self.show_next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # 图片展示区域
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.grid(row=1, column=0, sticky=tk.W + tk.E)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()

        # 状态信息
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.grid(row=2, column=0, sticky=tk.W + tk.E, pady=10)

        # 加载 YOLO 模型
        self.model1 = YOLO("1.pt")  # 替换成你的 YOLOv8 模型路径
        self.model2 = YOLO("2.pt")  # 替换成你的 YOLOv8 模型路径

        # 文件夹路径
        self.image_folder = 'D:/6.15/待测试文件夹'
        self.crops_folder = 'D:/6.15/最终检测/picture'
        self.save_folder2 = 'D:/6.15/最终检测/dic'

        # 图片列表和当前图片索引
        self.processed_images = []
        self.current_image_index = 0

    def start_processing(self):
        # 启动图片处理流程
        self.process_images()

        # 加载并展示处理后的图片
        self.load_processed_images()

        # 启用上一张/下一张按钮
        if self.processed_images:
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
            self.show_image(0)

    def process_images(self):
        # 第一段代码的逻辑：裁剪图片并保存到crops_folder
        if not os.path.exists(self.crops_folder):
            os.makedirs(self.crops_folder)

        image_files = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if
                       f.endswith(('jpg', 'png', 'jpeg', 'bmp'))]

        for image_file in image_files:
            im0 = cv2.imread(image_file)
            if im0 is None:
                print(f"Warning: {image_file} is not a valid image and will be skipped.")
                continue

            results = self.model1.predict(source=image_file, show=False)

            if isinstance(results, list):
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()

                    for idx, (box, cls) in enumerate(zip(boxes, clss)):
                        x1, y1, x2, y2 = box
                        crop_img = im0[int(y1):int(y2), int(x1):int(x2)]

                        save_path = os.path.join(self.crops_folder,
                                                 f"{os.path.basename(image_file).split('.')[0]}_{idx}_{int(cls)}.png")
                        cv2.imwrite(save_path, crop_img)

        # 第二段代码的逻辑：检测裁剪后的图片并保存到save_folder2
        if not os.path.exists(self.save_folder2):
            os.makedirs(self.save_folder2)

        for filename in os.listdir(self.crops_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.crops_folder, filename)
                image = cv2.imread(image_path)
                results = self.model2(image)

                detections = results[0].boxes
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{self.model2.names[box.cls[0].item()]} {box.conf[0].item():.2f}"
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                save_path = os.path.join(self.save_folder2, filename)
                cv2.imwrite(save_path, image)

        print("Image processing completed.")

    def load_processed_images(self):
        # 加载处理后的图片路径
        self.processed_images = [os.path.join(self.save_folder2, f) for f in os.listdir(self.save_folder2) if
                                 f.endswith(('jpg', 'png', 'jpeg'))]

        # 更新状态信息
        if self.processed_images:
            self.status_label.config(text=f"Loaded {len(self.processed_images)} processed images.")
        else:
            self.status_label.config(text="No processed images found.")

    def show_image(self, index):
        if 0 <= index < len(self.processed_images):
            self.current_image_index = index
            image_path = self.processed_images[index]

            # 加载图片并调整大小以适应显示区域
            img = Image.open(image_path)
            img.thumbnail((800, 600))  # 调整图片大小
            img_tk = ImageTk.PhotoImage(img)

            # 更新图片标签
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk  # 保持对图片的引用，防止被垃圾回收

            # 更新状态信息
            self.status_label.config(
                text=f"Image {index + 1} of {len(self.processed_images)}: {os.path.basename(image_path)}")

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)

    def show_next_image(self):
        if self.current_image_index < len(self.processed_images) - 1:
            self.show_image(self.current_image_index + 1)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()