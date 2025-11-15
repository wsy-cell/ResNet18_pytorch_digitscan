
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QRadioButton, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt6.QtCore import Qt, QPoint

from PIL import Image, ImageDraw
import numpy as np

from core.preprocess import preprocess_custom_image
from core.predict_final import load_model, predict
from core.digit_string_recognizer import DigitStringRecognizer
from core.preprocess import preprocess_custom_image
from core.data_manager import DataManager

class HandwriteWidget(QWidget):
    """
    但使用 PyQt6 重写，并可嵌入主界面
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(450, 380)

        # --------------------
        # 主题：暗色 + Material 风格
        # --------------------
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #dddddd;
                font-size: 15px;
            }
            QPushButton {
                background-color: #3a3a3a;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QRadioButton {
                padding: 3px;
            }
        """)

        # --------------------
        # 内部 PIL 图像（固定大小 400×300）
        # --------------------
        self.pil_w = 400
        self.pil_h = 300
        self.image = Image.new("L", (self.pil_w, self.pil_h), 255)
        self.draw = ImageDraw.Draw(self.image)

        # PyQt 显示画布
        self.canvas = QLabel()
        self.canvas.setFixedSize(self.pil_w, self.pil_h)
        self.canvas.setStyleSheet("background:white;")

        # 用于鼠标绘图
        self.last_point = None

        # 单/多数字模式
        self.rb_single = QRadioButton("单数字")
        self.rb_multi = QRadioButton("多数字串")
        self.rb_single.setChecked(True)

        # 按钮
        self.btn_predict = QPushButton("识别")
        self.btn_clear = QPushButton("清空")

        # 绑定事件
        self.btn_clear.clicked.connect(self.clear)
        self.btn_predict.clicked.connect(self.run_predict)

        # 主布局
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.rb_single)
        radio_layout.addWidget(self.rb_multi)
        layout.addLayout(radio_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_predict)
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.update_canvas()

    # -----------------------------------------------------------
    # 绘图相关
    # -----------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.pos()
            self.draw_point(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.draw_line(self.last_point, event.pos())
            self.last_point = event.pos()

    def draw_point(self, pos):
        if not self.is_inside_canvas(pos):
            return

        cx, cy = self.to_canvas_coords(pos)
        px, py = self.to_pil_coords(cx, cy)
        r = 12

        self.draw.ellipse((px-r, py-r, px+r, py+r), fill=0)
        self.update_canvas()

    def draw_line(self, p1, p2):
        if (not self.is_inside_canvas(p1)) and (not self.is_inside_canvas(p2)):
            return

        c1 = self.to_canvas_coords(p1)
        c2 = self.to_canvas_coords(p2)

        p1 = self.to_pil_coords(*c1)
        p2 = self.to_pil_coords(*c2)

        self.draw.line([p1, p2], fill=0, width=20)
        self.update_canvas()

    def is_inside_canvas(self, pos):
        x = pos.x()
        y = pos.y()
        w = self.canvas.width()
        h = self.canvas.height()
        return (0 <= x <= w) and (0 <= y <= h)

    def to_canvas_coords(self, pos):
        return pos.x(), pos.y()

    def to_pil_coords(self, cx, cy):
        px = int(cx * self.pil_w / self.canvas.width())
        py = int(cy * self.pil_h / self.canvas.height())
        return px, py

    # -----------------------------------------------------------
    # 更新画布显示
    # -----------------------------------------------------------
    def update_canvas(self):
        img = self.image.convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(qimg))

    # -----------------------------------------------------------
    # 逻辑功能
    # -----------------------------------------------------------
    def clear(self):
        self.image = Image.new("L", (self.pil_w, self.pil_h), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.update_canvas()

    def run_predict(self):
        
        model = load_model()
        dsr = DigitStringRecognizer(model)
        dm = DataManager()
        processed = preprocess_custom_image(self.image)

        if self.rb_single.isChecked():
            pred, _ = predict(model, processed)

            ok, label = dm.ask_user_confirm(pred, processed)
            if ok:
                dm.save_sample(processed, label)

            QMessageBox.information(self, "识别结果", f"数字：{pred}")

        else:  # 多数字串模式
            string, _ = dsr.recognize_string(self.image)
            QMessageBox.information(self, "识别结果", f"数字串：{string}")

            # 保存分割样本
            digit_imgs = dsr.segment_digits(self.image)
            for img, digit in zip(digit_imgs, string):
                ok, label = dm.ask_user_confirm(int(digit), img)
                if ok:
                    dm.save_sample(img, label)
