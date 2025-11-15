# ============================================================
# 增量微调（Incremental Finetune）
#
# 用途：
#   - 在已有模型基础上，使用 ./data/custom 的新增样本进行快速微调
#   - 保存新的模型版本（由 core.model_manager 管理）
#
# 要点：
#   - 优先使用 models/ 中的最新已保存模型作为基准；若无模型则构建新模型
#   - 冻结除最后全连接层外的参数（可选），只训练 FC 层以加快微调
#   - 微调后通过 core.model_manager.save_model 保存新版本
# ============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from utils.logger import info, warn, error, debug
from core.model_manager import list_versions, load_version, save_model
from core.train_final import preprocess_custom_image, get_resnet18  # 直接复用 train_final 中的预处理与模型构造


# -------------------------
# 自定义 Dataset（代替 ImageFolder，强制灰度 + preprocess）
# -------------------------
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        if not os.path.isdir(root):
            return
        for label in os.listdir(root):
            folder = os.path.join(root, label)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(folder, f), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("L")
        img = preprocess_custom_image(img)
        if self.transform:
            img = self.transform(img)
        return img, label


# -------------------------
# 构建 custom + mnist 的 loader（和 full train 保持一致的 normalize）
# -------------------------
def build_custom_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    datasets = []
    # 保留 MNIST 以稳定训练（可选）
    mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    datasets.append(mnist)

    if os.path.isdir("./data/custom"):
        custom = CustomDataset("./data/custom", transform=transform)
        datasets.append(custom)
        info(f"已加载自定义数据集：{len(custom)} 张")
    else:
        warn("未找到 ./data/custom，自定义微调将只使用 MNIST（若需要请放入样本）。")

    concat = ConcatDataset(datasets)
    loader = DataLoader(concat, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


# -------------------------
# 查找最新版本模型（若存在）
# -------------------------
def load_latest_model(device=None):
    meta = list_versions()
    if not meta:
        info("models/ 目录中无已保存模型，使用新模型初始化")
        model = get_resnet18()
        if device:
            model.to(device)
        return model

    # meta 是按保存顺序追加的，取最后一项作为最新
    latest_entry = meta[-1]
    path = latest_entry["path"]
    info(f"加载最新模型版本：{path}")
    state = torch.load(path, map_location="cpu")
    model = get_resnet18()
    model.load_state_dict(state)
    if device:
        model.to(device)
    return model


# -------------------------
# 主函数：增量微调
# -------------------------
def finetune(last_n_epochs=5, lr=3e-4, batch_size=64, device=None):
    info("开始增量微调（Incremental Finetune）")

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    loader = build_custom_loader(batch_size=batch_size)

    # 加载最新模型（如果存在）
    model = load_latest_model(device=device)

    # 冻结除 FC 外的参数（常用的快速微调策略）
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    best_acc = 0.0

    for epoch in range(last_n_epochs):
        info(f"==== 第 {epoch+1}/{last_n_epochs} 轮微调开始 ====")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, preds = out.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1) * 100.0
        info(f"微调：loss={epoch_loss:.4f}  acc={epoch_acc:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy_state_dict(model.state_dict())
            best_acc = epoch_acc

    # 保存微调后的最佳模型（由 model_manager 统一命名与记录）
    entry = save_model(best_state or model.state_dict(), val_acc=best_acc, notes="incremental_finetune")
    info(f"✔ 微调完成并已保存为新版本：{entry['path']} (val_acc={best_acc:.2f}%)")


# 辅助：深拷贝 state_dict
def copy_state_dict(sd):
    return {k: v.cpu().clone() for k, v in sd.items()}
