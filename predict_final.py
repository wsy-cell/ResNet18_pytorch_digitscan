import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms

# ============== 日志系统 ==============
from utils.logger import info, error, debug, warn


# ===================================================
# 1. 预处理（与训练一致）
# ===================================================
def preprocess(img):
    info("开始对图像进行预处理")

    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.GaussianBlur(1))

    arr = np.array(img)
    ys, xs = np.where(arr < 255)
    if len(xs) > 0:
        left, right = xs.min(), xs.max()
        top, bottom = ys.min(), ys.max()
        arr = arr[top:bottom+1, left:right+1]

    img = Image.fromarray(arr).resize((28,28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    info("图像预处理完成")
    return transform(img).unsqueeze(0)


# ===================================================
# 2. 加载模型
# ===================================================
def load_model(weight="models/default_model.pth"):
    info(f"加载模型：{weight}")
    
     # 打印调用栈
    import traceback
    traceback.print_stack()
    
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1,64,3,1,1,bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)

    try:
        model.load_state_dict(torch.load(weight, map_location="cpu"))
        info(f"✔ 模型加载成功：{weight}")
    except Exception as e:
        error(f"加载模型失败：{e}")
        raise e

    model.eval()
    return model


# ===================================================
# 3. 推理函数（供GUI/摄像头调用）
# ===================================================
def predict(model, img: Image.Image):
    info("开始进行图像预测")

    x = preprocess(img)

    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)
        pred = prob.argmax().item()

    info(f"预测结果：{pred}，预测概率：{prob.numpy().tolist()}")
    return pred, prob.numpy().tolist()
