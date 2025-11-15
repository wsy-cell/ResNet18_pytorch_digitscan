import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y-%m-%d')}.log")

logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ========== UI 日志桥 ==========
_ui_callbacks = []

def bind_ui(callback):
    """允许 UI（QTextEdit.append）绑定"""
    _ui_callbacks.append(callback)

def log_ui(msg):
    """把日志同步给 UI 控件"""
    for cb in _ui_callbacks:
        cb(msg)


# ========== 对外统一 API ==========
def info(msg):
    logger.info(msg)
    log_ui(msg)

def error(msg):
    logger.error(msg)
    log_ui("X" + msg)

def debug(msg):
    logger.debug(msg)
    log_ui("[DEBUG] " + msg)

def warn(msg):
    logger.warning(msg)
    log_ui("⚠ " + msg)
