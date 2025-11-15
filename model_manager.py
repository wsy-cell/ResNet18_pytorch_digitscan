import os
import json
import time
from pathlib import Path
import torch

ROOT = Path("./models")  # 模型保存的统一路径
META = ROOT / "metadata.json"
ROOT.mkdir(parents=True, exist_ok=True)

def _load_meta():
    """加载模型版本的元数据"""
    if not META.exists():
        return []
    with open(META, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_meta(meta):
    """保存模型版本的元数据"""
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def save_model(state_dict, val_acc=None, notes=""):
    """保存模型并记录元数据"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_model.pth"
    path = ROOT / fname
    torch.save(state_dict, str(path))

    meta = _load_meta()
    entry = {
        "time": ts,
        "path": str(path),
        "val_acc": float(val_acc) if val_acc is not None else None,
        "notes": notes
    }
    meta.append(entry)
    _save_meta(meta)
    return entry

def list_versions():
    """列出所有已保存的模型版本"""
    return _load_meta()

def load_version(index):
    """加载指定版本的模型"""
    meta = _load_meta()
    if index < 0 or index >= len(meta):
        raise IndexError("版本索引越界")
    entry = meta[index]
    path = entry["path"]
    state = torch.load(path, map_location="cpu")
    return state, entry
