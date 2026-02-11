import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os


class ImageDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片展示界面")

        # 设置窗口大小
        self.root.geometry("800x600")

        # 创建界面元素
        self.create_widgets()

    def create_widgets(self):
        # 创建一个标签用于显示图片
        self.display_label = tk.Label(self.root, text="请选择图片进行展示", bg="white")
        self.display_label.pack(pady=20, padx=20, fill="both", expand=True)

        # 创建按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # 创建选择文件按钮
        self.select_button = tk.Button(button_frame, text="选择图片文件", command=self.select_image)
        self.select_button.pack(side="left", padx=10)

        # 创建开始展示按钮
        self.start_button = tk.Button(button_frame, text="开始展示", command=self.start_display)
        self.start_button.pack(side="left", padx=10)

    def select_image(self):
        # 打开文件对话框选择图片
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )

        if file_path:
            self.image_path = file_path
            self.start_button.config(state="normal")  # 启用开始展示按钮
            self.load_image()  # 预览图片

    def load_image(self):
        # 加载并显示图片
        try:
            # 打开图片并调整大小以适应标签
            image = Image.open(self.image_path)
            # 调整图片大小以适合显示区域
            image.thumbnail((700, 400))
            self.photo = ImageTk.PhotoImage(image)

            # 更新标签显示图片
            self.display_label.config(image=self.photo, text="")
        except Exception as e:
            self.display_label.config(text=f"加载图片失败: {e}", image="")
            self.start_button.config(state="disabled")

    def start_display(self):
        # 开始展示图片
        self.load_image()  # 确保图片已加载
        self.display_label.config(text="")  # 清除提示文字
        # 这里可以添加更多展示逻辑，如幻灯片效果等


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDisplayApp(root)
    root.mainloop()