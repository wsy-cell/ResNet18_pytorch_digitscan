# UI/sidebar.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from .style import SIDEBAR_STYLE


class Sidebar(QWidget):
    page_changed = pyqtSignal(int)  # 点击触发切换页面

    def __init__(self):
        super().__init__()
        self.setFixedWidth(180)
        self.setStyleSheet("background-color: #ffffff;")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(10, 20, 10, 20)

        self.buttons = []

        items = [
            ("📷 摄像头识别", 0),
            ("📁 模型管理", 1),
            ("✍️ 手写识别", 2),
            ("🚀 模型训练", 3),
        ]

        for text, idx in items:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, x=idx: self.set_page(x))
            btn.setStyleSheet(SIDEBAR_STYLE)
            layout.addWidget(btn)
            self.buttons.append(btn)

        layout.addStretch()
        self.setLayout(layout)

        # 默认选中第0页
        self.buttons[0].setChecked(True)

    def set_page(self, index):
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)
        self.page_changed.emit(index)
