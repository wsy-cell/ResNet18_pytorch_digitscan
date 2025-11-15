import threading
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QFrame
from PyQt6.QtCore import QThread, pyqtSignal
from utils.logger import info, error, warn
from .style import CARD_STYLE

# 摄像头实时识别处理类
class CameraPage(QWidget):
    def __init__(self):
        super().__init__()

        card = QFrame()
        card.setStyleSheet(CARD_STYLE)

        layout = QVBoxLayout()

        self.btn_start = QPushButton("启动摄像头")
        self.btn_stop = QPushButton("停止摄像头")
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.log)

        card.setLayout(layout)

        self.setLayout(layout)

        # 日志界面绑定
        info("摄像头页面初始化完成")

        # 摄像头启动/停止按钮
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)

        # 默认停止状态
        self.is_running = False
        self.camera_runner = None

    def start_camera(self):
        """启动摄像头并开始实时识别"""
        if self.is_running:
            info("摄像头已在运行")
            return

        info("启动摄像头识别...")
        self.camera_runner = CameraRunner(self.update_log)
        self.camera_runner.start()
        self.is_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_camera(self):
        """停止摄像头识别"""
        if not self.is_running:
            info("摄像头未在运行")
            return

        info("停止摄像头识别...")
        self.camera_runner.stop()
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_log(self, log_message):
        """更新日志框"""
        self.log.append(log_message)


class CameraRunner(QThread):
    update_log = pyqtSignal(str)

    def __init__(self, update_log_callback):
        super().__init__()
        self.update_log_callback = update_log_callback
        self._stop = threading.Event()
        self._running = False

    def run(self):
        """摄像头捕捉和识别线程"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            error("无法打开摄像头")
            return

        self._running = True
        info("摄像头开启，开始捕捉视频流")

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                warn("摄像头读取失败")
                break

            # 处理图像，缩放帧
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 模拟实时识别过程
            # 你可以替换成调用你的识别函数
            label, prob = self.predict(gray)

            # 在屏幕上显示识别结果
            cv2.putText(frame, f"Pred: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Camera - Preview", frame)

            # 将识别信息传递到日志框
            log_message = f"预测：{label}, 概率：{prob}"
            self.update_log.emit(log_message)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._running = False
        info("摄像头线程结束")

    def predict(self, gray_image):
        """模拟预测过程"""
        # 这里的模拟逻辑可以替换为实际的手写数字识别模型预测
        # 假设预测结果为数字 5，概率为 98%
        return 5, 0.98
