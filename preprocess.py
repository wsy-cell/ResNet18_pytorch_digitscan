import numpy as np
from PIL import ImageOps, Image, ImageFilter

def preprocess_custom_image(img):
    """
    将任意图像（画板/摄像头/照片）转换为 MNIST 风格的 28×28 单通道图像。
    """
    img = img.convert("L")

    # 二值化
    img = img.point(lambda p: 0 if p < 128 else 255)

    # 白底黑字 → 黑底白字
    img = ImageOps.invert(img)

    # 高斯模糊（模拟 MNIST 风格）
    img = img.filter(ImageFilter.GaussianBlur(1))

    # 取出有效笔迹区域（自动居中）
    arr = np.array(img)
    ys, xs = np.where(arr < 255)

    if len(xs) > 0:
        left, right = xs.min(), xs.max()
        top, bottom = ys.min(), ys.max()
        arr = arr[top:bottom+1, left:right+1]

    img = Image.fromarray(arr)

    # 统一缩放
    img = img.resize((28, 28))

    return img
