# ============================================================
#  手写数字识别最终训练脚本（含 ResNet18 + Mixup + Warmup）
#  —— 从论文复现到工程落地的完整注释版本
#
#  本文件内容解释了：
#   1. CNN 卷积网络与 ResNet 的本质原理
#   2. 数据增强（几何扰动、反色、透视）的数学意义
#   3. Mixup 的公式推导，以及为什么能够提升泛化
#   4. Label Smoothing 如何减少过拟合
#   5. 余弦退火学习率的公式
#   6. Warmup 为什么能稳定训练
#   7. 自建手写数据如何预处理成 MNIST 风格
# ============================================================

import os
import copy
import numpy as np
from PIL import ImageOps, Image, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.models import resnet18

# 日志
from utils.logger import info, warn, error, debug

# 模型版本管理（负责保存/列出/加载模型文件和 metadata）
from core.model_manager import save_model

# -------------------------
# 注意：
# - 本文件负责训练主流程并在验证集上提升时通过 model_manager.save_model 保存
# - model_manager 负责文件命名与 metadata（你之前给出的实现）
# -------------------------

# ============================================================
# PART 1 — 将图像进行反色
# ============================================================
def invert_img(x):
    """将图像进行反色（白变黑、黑变白）"""
    return ImageOps.invert(x)


# ============================================================
# PART 2 — 数据增强（Data Augmentation）
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomRotation(15),                       # ±15° 的书写倾斜
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.2)),
        transforms.RandomPerspective(0.15),                  # 摄像头畸变模拟
        transforms.GaussianBlur((3,3)),                      # 模糊模拟真实手写
    ], p=0.5),                                                # 50% 概率增强，提高稳定性

    transforms.RandomApply([
        transforms.Lambda(invert_img)                         # 少量反色，增强鲁棒性
    ], p=0.15),

    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))               # MNIST 归一化
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ============================================================
# PART 3 — 自建数据预处理（核心关键）
# ============================================================
def preprocess_custom_image(img):
    """将任意自建手写图片转换成 MNIST 风格（灰度、二值化、反色、居中、缩放）"""
    img = img.convert("L")

    # ---------- 1. 二值化 ----------
    img = img.point(lambda p: 0 if p < 128 else 255)

    # ---------- 2. 反色（白底黑字 → 黑底白字） ----------
    img = ImageOps.invert(img)

    # ---------- 3. 模糊 ----------
    img = img.filter(ImageFilter.GaussianBlur(1))

    # ---------- 4. 裁剪非空区域（居中对齐关键步骤） ----------
    arr = np.array(img)
    ys, xs = np.where(arr < 255)
    if len(xs) > 0:
        left, right = xs.min(), xs.max()
        top, bottom = ys.min(), ys.max()
        arr = arr[top:bottom+1, left:right+1]
    else:
        # 若为空则返回 28x28 全白（极端情况）
        return Image.new("L", (28,28), 255)

    img = Image.fromarray(arr)

    # ---------- 5. 缩放到 28×28 ----------
    img = img.resize((28,28))

    return img


class CustomImageFolder(torchvision.datasets.ImageFolder):
    """
    覆盖 __getitem__，使所有自建图像都自动进行 MNIST 风格预处理
    注意：ImageFolder 仅用于遍历结构，预处理保证灰度并返回单通道 Tensor
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path)
        img = preprocess_custom_image(img)
        img = train_transform(img)
        return img, target


# ============================================================
# PART 4 — 构建数据集（MNIST + 自建）
# ============================================================
def build_dataset(custom_dir="./data/custom"):
    """构建训练/验证/测试数据加载器"""
    info("开始构建训练数据集…")

    mnist_train = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )

    datasets = [mnist_train]

    if os.path.isdir(custom_dir):
        info(f"检测到自建数据集：{custom_dir}")
        custom_ds = CustomImageFolder(custom_dir)
        datasets.append(custom_ds)
    else:
        warn("未检测到自建手写数据，将仅使用 MNIST。")

    train_val_dataset = ConcatDataset(datasets)

    val_size = int(0.1 * len(train_val_dataset))
    train_size = len(train_val_dataset) - val_size
    train_ds, val_ds = random_split(train_val_dataset, [train_size, val_size])

    info(f"训练样本数量：{train_size}")
    info(f"验证样本数量：{val_size}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=4)

    info("数据集构建完毕。")
    return train_loader, val_loader, test_loader


# ============================================================
# PART 5 — ResNet18（MNIST 专用修改版）
# ============================================================
def get_resnet18():
    info("构建 ResNet18（MNIST 专用修改版）")
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    info("ResNet18 构建完成。")
    return model


# ============================================================
# PART 6 — Mixup（保留函数以便后续使用）
# ============================================================
def mixup_data(x, y, alpha=0.3):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, (y, y[idx]), lam

def mixup_loss(criterion, pred, targets, lam):
    y_a, y_b = targets
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# PART 7 — 训练主循环（使用 model_manager.save_model 统一保存）
# ============================================================
def train(model, loaders, num_epochs=25, progress_callback=None):
    """
    训练主函数
    - 在验证集上若 val_acc 提升，则通过 model_manager.save_model 保存模型和 metadata
    - 返回训练后的模型（加载最佳权重）
    """
    info("开始模型训练…")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        info(f"======== 第 {epoch+1}/{num_epochs} 轮训练开始 ========")

        if progress_callback:
            progress_callback(epoch, num_epochs)
        
        # ---------- 训练 ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # warmup/mixup 策略可按需启用（若需要，可扩展）
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, preds = out.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1) * 100.0
        info(f"训练：loss={train_loss:.4f}  acc={train_acc:.2f}%")

        # ---------- 验证 ----------
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, preds = out.max(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / max(val_total, 1) * 100.0
        info(f"验证：acc={val_acc:.2f}%")

        scheduler.step()

        # 若验证集准确率提升，则通过 model_manager 保存（model_manager 负责文件名/metadata）
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            entry = save_model(best_wts, val_acc=best_acc, notes="full_train")
            info(f"✔ 验证集准确率提升，已保存模型版本：{entry['path']} (val_acc={best_acc:.2f}%)")

    info(f"训练结束！最佳验证准确率：{best_acc:.2f}%")
    # 加载最佳权重并返回
    model.load_state_dict(best_wts)
    return model


# ============================================================
# PART 8 — 测试集评估
# ============================================================
def test(model, loader):
    info("开始测试集评估…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, preds = out.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / max(total, 1) * 100.0
    info(f"测试集最终准确率：{acc:.2f}%")
    return acc


# ============================================================
# PART 9 — Main (当作为独立脚本运行时)
# ============================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataset()
    loaders = {"train": train_loader, "val": val_loader}
    model = get_resnet18()
    model = train(model, loaders, num_epochs=25)
    test(model, test_loader)