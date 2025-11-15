from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QProgressBar, QPushButton,
    QFrame, QTextEdit, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal
from .style import CARD_STYLE
from utils.logger import info, error, warn

import traceback
import os

from core.train_final import build_dataset, get_resnet18, train as full_train
from core.incremental_train import finetune as incremental_train


class TrainPage(QWidget):
    """
    训练页面（完整训练 + 增量微调）
    """
    def __init__(self):
        super().__init__()

        card = QFrame()
        card.setStyleSheet(CARD_STYLE)

        layout = QVBoxLayout()

        # === 按钮 ===
        self.btn_full = QPushButton("开始训练（完整训练）")
        self.btn_inc = QPushButton("微调模型（增量训练）")

        # === 进度条 ===
        self.progress = QProgressBar()
        self.progress.setValue(0)

        # === 日志框 ===
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(self.btn_full)
        layout.addWidget(self.btn_inc)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        card.setLayout(layout)

        main = QVBoxLayout()
        main.addWidget(card)
        self.setLayout(main)

        # 信号连接
        self.btn_full.clicked.connect(self.start_full_train)
        self.btn_inc.clicked.connect(self.start_incremental_train)

        info("训练页面初始化完成")

    # ==========================================================
    # 启动完整训练
    # ==========================================================
    def start_full_train(self):
        info("开始完整训练")
        self._start_thread("full")

    # ==========================================================
    # 启动增量微调
    # ==========================================================
    def start_incremental_train(self):
        info("开始增量微调训练")
        self._start_thread("incremental")

    # ==========================================================
    # 公共启动线程函数
    # ==========================================================
    def _start_thread(self, mode):
        self.progress.setValue(0)
        self.log.clear()

        self.thread = TrainThread(mode)
        self.thread.update_log.connect(self.append_log)
        self.thread.update_progress.connect(self.progress.setValue)
        self.thread.finished.connect(self._train_finished)

        self.btn_full.setEnabled(False)
        self.btn_inc.setEnabled(False)

        self.thread.start()

    def append_log(self, msg):
        self.log.append(msg)

    def _train_finished(self):
        self.btn_full.setEnabled(True)
        self.btn_inc.setEnabled(True)
        QMessageBox.information(self, "完成", "训练已结束！")
        info("训练线程结束")


# ==============================================================
# 后台训练线程
# ==============================================================

class TrainThread(QThread):
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(self, mode):
        super().__init__()
        self.mode = mode  # "full" 或 "incremental"

    def _emit(self, msg):
        """辅助：更新日志并写入logger"""
        self.update_log.emit(msg)
        info(msg)

    def run(self):
        try:
            if self.mode == "full":
                self._run_full_train()

            elif self.mode == "incremental":
                self._run_incremental()

        except Exception as e:
            err_msg = f"训练时发生错误：\n{e}\n{traceback.format_exc()}"
            error(err_msg)
            self.update_log.emit(err_msg)

    # ==========================================================
    # 完整训练流程
    # ==========================================================
    def _run_full_train(self):
        self._emit("【完整训练】开始构建数据集…")
        train_loader, val_loader, test_loader = build_dataset()
        loaders = {"train": train_loader, "val": val_loader}

        self.update_progress.emit(10)

        self._emit("【完整训练】构建 ResNet18 模型…")
        model = get_resnet18()
        self.update_progress.emit(20)

        # 回调给用户看训练过程
        def report(epoch, total_epochs):
            pct = int((epoch / total_epochs) * 100)
            self.update_progress.emit(20 + pct)

        self._emit("【完整训练】开始训练…")
        model = full_train(model, loaders, num_epochs=25, progress_callback=report)

        self.update_progress.emit(100)
        self._emit("【完整训练】训练完成！")

    # ==========================================================
    # 增量微调训练
    # ==========================================================
    def _run_incremental(self):
        self._emit("【增量微调】开始检查 data/custom…")

        # 自定义数据数量统计
        total = 0
        if os.path.isdir("./data/custom"):
            for d in os.listdir("./data/custom"):
                p = os.path.join("./data/custom", d)
                if os.path.isdir(p):
                    total += len(os.listdir(p))

        self._emit(f"【增量微调】发现自建样本 {total} 张")

        self.update_progress.emit(10)

        self._emit("【增量微调】开始 finetune()…")
        incremental_train(last_n_epochs=5)

        self.update_progress.emit(100)
        self._emit("【增量微调】微调完成！")
