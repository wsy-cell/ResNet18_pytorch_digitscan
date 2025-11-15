import os
from PIL import Image

class DataManager:
    def __init__(self, base_dir="./data/custom"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_sample(self, img, label):
        """
        img: PIL 或 (x, PIL)
        自动转换为 28×28 灰度图
        """
        # 自动从 (x, img) 取出 PIL 图像
        if isinstance(img, tuple):
            img = img[1]

        if not hasattr(img, "convert"):
            raise ValueError(f"[DataManager] save_sample 接收到非图像对象: {type(img)}")

        # 强制转换为灰度图 L
        img = img.convert("L")

        # 强制统一缩放尺寸到 28x28（和 MNIST 一样）
        img = img.resize((28, 28))

        folder = os.path.join(self.base_dir, str(label))
        os.makedirs(folder, exist_ok=True)

        idx = len(os.listdir(folder))
        path = os.path.join(folder, f"{idx}.png")

        img.save(path)
        print(f"[DataManager] 已保存灰度样本: {path}")


    def ask_user_confirm(self, pred, img):
        import tkinter as tk
        from tkinter import messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()

        ok = messagebox.askyesno("确认识别结果", f"识别为 {pred}，是否正确？")
        if ok:
            return True, pred

        label = simpledialog.askinteger("输入正确标签", "请输入正确标签 (0-9)：")
        if label is None:
            return False, None

        if 0 <= label <= 9:
            return True, label
        else:
            messagebox.showerror("错误", "标签必须为 0-9！")
            return False, None
