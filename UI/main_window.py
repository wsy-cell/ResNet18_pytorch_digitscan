# UI/main_window.py

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QStackedWidget
from .sidebar import Sidebar
from .style import GLOBAL_STYLE

from .page_camera import CameraPage
from .page_model_manager import ModelManagerPage
from .page_handwrite import HandwritePage
from .page_train import TrainPage


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("手写数字识别系统（侧边栏版）")
        self.setMinimumSize(1180, 700)
        self.setStyleSheet(GLOBAL_STYLE)

        # 左侧导航栏
        self.sidebar = Sidebar()

        # 右侧切换页面
        self.pages = QStackedWidget()
        self.page_camera = CameraPage()
        self.page_manage = ModelManagerPage()
        self.page_handwrite = HandwritePage()
        self.page_train = TrainPage()

        self.pages.addWidget(self.page_camera)
        self.pages.addWidget(self.page_manage)
        self.pages.addWidget(self.page_handwrite)
        self.pages.addWidget(self.page_train)

        # 布局
        layout = QHBoxLayout()
        layout.addWidget(self.sidebar)
        layout.addWidget(self.pages)
        self.setLayout(layout)

        # 绑定页面切换
        self.sidebar.page_changed.connect(self.pages.setCurrentIndex)
