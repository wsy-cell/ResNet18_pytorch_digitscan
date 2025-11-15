import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTextEdit, QFrame,
    QListWidget, QListWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt

from utils.logger import info, warn, error
from .style import CARD_STYLE

from core.model_manager import list_versions, load_version


class ModelManagerPage(QWidget):
    """
    模型管理页面：
    - 列出 metadata.json 中的所有版本
    - 展示保存时间 / 验证准确率 / 备注 等信息
    - 支持点击加载某一版本
    - 与训练脚本（train_final, incremental_train）完全一致
    """

    def __init__(self):
        super().__init__()

        # 外层卡片
        card = QFrame()
        card.setStyleSheet(CARD_STYLE)

        card_layout = QVBoxLayout()

        # ------- 模型版本列表 -------
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        # 按钮
        self.refresh_button = QPushButton("刷新模型列表")
        self.load_button = QPushButton("加载所选模型")

        # 日志显示
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # 加入布局
        card_layout.addWidget(self.model_list)
        card_layout.addWidget(self.refresh_button)
        card_layout.addWidget(self.load_button)
        card_layout.addWidget(self.log)

        card.setLayout(card_layout)

        main = QVBoxLayout()
        main.addWidget(card)
        self.setLayout(main)

        # 绑定按钮事件
        self.refresh_button.clicked.connect(self.refresh_model_list)
        self.load_button.clicked.connect(self.load_selected_model)

        # 初始化页面
        info("模型管理页面初始化完成")
        self.log.append("模型管理页面初始化完成。")
        self.refresh_model_list()

        # 当前加载的模型权重（提供给主程序调用）
        self.current_model_state = None
        self.current_meta = None


    # ============================================================
    # 刷新模型列表
    # ============================================================
    def refresh_model_list(self):
        """重新读取 metadata.json并刷新 UI"""
        self.model_list.clear()
        self.log.append("正在扫描模型版本...")

        try:
            versions = list_versions()  # 从 model_manager 获取
        except Exception as e:
            error(f"读取模型 metadata 失败：{e}")
            self.log.append(f"读取 metadata 失败：{e}")
            return

        if not versions:
            warn("未检测到任何模型版本。")
            self.log.append("未找到模型版本。")
            return

        # 逐条加入列表显示
        for idx, item in enumerate(versions):
            time_str = item.get("time", "未知时间")
            acc = item.get("val_acc", None)
            notes = item.get("notes", "")

            text = f"[{idx}]  时间: {time_str}   准确率: {acc}   备注: {notes}"

            list_item = QListWidgetItem(text)
            list_item.setData(Qt.ItemDataRole.UserRole, idx)   # 绑定版本索引
            self.model_list.addItem(list_item)

        info("模型版本列表刷新完毕。")
        self.log.append("模型列表刷新完毕。")


    # ============================================================
    #  加载选中模型
    # ============================================================
    def load_selected_model(self):
        """加载 UI 中选中的模型版本（真实加载，不模拟）"""

        item = self.model_list.currentItem()
        if not item:
            warn("未选择模型版本")
            self.log.append("请先选择一个模型版本。")
            return

        index = item.data(Qt.ItemDataRole.UserRole)

        try:
            state_dict, meta = load_version(index)
        except Exception as e:
            error(f"模型加载失败：{e}")
            QMessageBox.critical(self, "错误", f"加载失败：{e}")
            return

        # 存入属性（提供给主程序 / 主窗口调用）
        self.current_model_state = state_dict
        self.current_meta = meta

        info(f"模型加载成功：{meta['path']}")
        self.log.append(f"✔ 加载成功：{meta['path']}")

        QMessageBox.information(self, "模型加载成功", f"已加载版本：{meta['time']}")


    # ============================================================
    # 主程序可调用接口（例如 Camera / Handwrite 使用）
    # ============================================================
    def get_loaded_model_state(self):
        """返回当前加载的模型 state_dict"""
        return self.current_model_state

    def get_loaded_meta(self):
        """返回当前版本元数据"""
        return self.current_meta
