# from PyQt6.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
#     QTextEdit, QLabel, QFrame, QRadioButton, QMessageBox
# )
# from PyQt6.QtGui import QPixmap, QImage
# from PyQt6.QtCore import Qt
# import numpy as np
# from PIL import Image, ImageDraw

# from utils.logger import info, error
# from .style import CARD_STYLE

# # 保留你原来的 core 模块
# from core.preprocess import preprocess_custom_image
# from core.predict_final import load_model, predict
# from core.digit_string_recognizer import DigitStringRecognizer
# from core.data_manager import DataManager


# class HandwritePage(QWidget):
#     """
#     将手写画板、按钮、识别、日志全部集中到一个页面内
#     """

#     def __init__(self):
#         super().__init__()

#         # =========================
#         # 初始化模型与数据管理器
#         # =========================
#         self.model = load_model()
#         self.dsr = DigitStringRecognizer(self.model)
#         self.dm = DataManager()

#         # =========================
#         # 画布基础设置
#         # =========================
#         self.pil_w, self.pil_h = 400, 300
#         self.image = Image.new("L", (self.pil_w, self.pil_h), 255)
#         self.draw = ImageDraw.Draw(self.image)
#         self.last_point = None

#         # =========================
#         # 外层卡片
#         # =========================
#         card = QFrame()
#         card.setStyleSheet(CARD_STYLE)

#         main_layout = QVBoxLayout(card)

#         # =========================
#         # 显示画布（QLabel）
#         # =========================
#         self.canvas = QLabel()
#         self.canvas.setFixedSize(self.pil_w, self.pil_h)
#         self.canvas.setStyleSheet("background:white;")
#         main_layout.addWidget(self.canvas)

#         # 更新显示
#         self.update_canvas()

#         # =========================
#         # 单/多数字模式
#         # =========================
#         mode_layout = QHBoxLayout()
#         self.rb_single = QRadioButton("单数字")
#         self.rb_multi = QRadioButton("多数字串")
#         self.rb_single.setChecked(True)
#         mode_layout.addWidget(self.rb_single)
#         mode_layout.addWidget(self.rb_multi)
#         main_layout.addLayout(mode_layout)

#         # =========================
#         # 按钮区
#         # =========================
#         btn_layout = QHBoxLayout()
#         self.btn_recognize = QPushButton("开始识别")
#         self.btn_clear = QPushButton("清空")

#         btn_layout.addWidget(self.btn_recognize)
#         btn_layout.addWidget(self.btn_clear)
#         main_layout.addLayout(btn_layout)

#         # 绑定事件
#         self.btn_clear.clicked.connect(self.clear)
#         self.btn_recognize.clicked.connect(self.recognize)

#         # =========================
#         # 日志区域
#         # =========================
#         self.log = QTextEdit()
#         self.log.setReadOnly(True)
#         main_layout.addWidget(self.log)

#         # 设置最终布局
#         self.setLayout(main_layout)

#         info("整合后的手写界面初始化完成")

#     # -----------------------------------------------------------
#     # 绘图事件
#     # -----------------------------------------------------------
#     def mousePressEvent(self, event):
#         if event.button() == Qt.MouseButton.LeftButton:
#             self.last_point = event.pos()
#             self.draw_point(event.pos())

#     def mouseMoveEvent(self, event):
#         if event.buttons() & Qt.MouseButton.LeftButton:
#             self.draw_line(self.last_point, event.pos())
#             self.last_point = event.pos()

#     def is_inside_canvas(self, pos):
#         x, y = pos.x(), pos.y()
#         return 0 <= x <= self.canvas.width() and 0 <= y <= self.canvas.height()

#     def to_pil(self, pos):
#         px = int(pos.x() * self.pil_w / self.canvas.width())
#         py = int(pos.y() * self.pil_h / self.canvas.height())
#         return px, py

#     def draw_point(self, pos):
#         if not self.is_inside_canvas(pos):
#             return
#         px, py = self.to_pil(pos)
#         r = 12
#         self.draw.ellipse((px-r, py-r, px+r, py+r), fill=0)
#         self.update_canvas()

#     def draw_line(self, p1, p2):
#         if not self.is_inside_canvas(p1) and not self.is_inside_canvas(p2):
#             return
#         p1 = self.to_pil(p1)
#         p2 = self.to_pil(p2)
#         self.draw.line([p1, p2], fill=0, width=20)
#         self.update_canvas()

#     # -----------------------------------------------------------
#     # 更新画布
#     # -----------------------------------------------------------
#     def update_canvas(self):
#         img = self.image.convert("RGB")
#         arr = np.array(img)
#         h, w, _ = arr.shape
#         qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
#         self.canvas.setPixmap(QPixmap.fromImage(qimg))

#     # -----------------------------------------------------------
#     # 清空
#     # -----------------------------------------------------------
#     def clear(self):
#         self.image = Image.new("L", (self.pil_w, self.pil_h), 255)
#         self.draw = ImageDraw.Draw(self.image)
#         self.update_canvas()
#         self.log.append("已清空画布\n")

#     # -----------------------------------------------------------
#     # 识别功能
#     # -----------------------------------------------------------
#     def recognize(self):
#         try:
#             if self.rb_single.isChecked():
#                 self.do_single_digit()
#             else:
#                 self.do_multi_digit()

#         except Exception as e:
#             error(f"识别失败：{e}")
#             self.log.append(f"识别失败：{e}\n")

#     # -----------------------------------------------------------
#     # 单数字模式
#     # -----------------------------------------------------------
#     def do_single_digit(self):
#         processed = preprocess_custom_image(self.image)
#         pred, prob = predict(self.model, processed)

#         ok, label = self.dm.ask_user_confirm(pred, processed)
#         if ok:
#             self.dm.save_sample(processed, label)

#         self.log.append(f"单数字识别 → {pred}    概率：{prob:.4f}\n")
#         QMessageBox.information(self, "识别结果", f"数字：{pred}")

#     # -----------------------------------------------------------
#     # 多数字模式
#     # -----------------------------------------------------------
#     def do_multi_digit(self):
#         string, prob = self.dsr.recognize_string(self.image)
#         digit_imgs = self.dsr.segment_digits(self.image)

#         self.log.append(f"多数字串识别 → {string}\n")

#         # 保存每个数字
#         for img, d in zip(digit_imgs, string):
#             ok, label = self.dm.ask_user_confirm(int(d), img)
#             if ok:
#                 self.dm.save_sample(img, label)

#         QMessageBox.information(self, "识别结果", f"数字串：{string}")
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QFrame, QMessageBox, QSlider
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QRect
import numpy as np
from PIL import Image, ImageDraw

from utils.logger import info, error
from .style import CARD_STYLE

# core 模块保持不动
from core.preprocess import preprocess_custom_image
from core.predict_final import load_model
from core.digit_string_recognizer import DigitStringRecognizer
from core.data_manager import DataManager


class HandwritePage(QWidget):
    """
    支持：撤销、橡皮擦、粗细调节、等比例缩放的新版手写面板
    """

    def __init__(self):
        super().__init__()

        # =========================
        # 模型加载
        # =========================
        self.model = load_model()
        self.dsr = DigitStringRecognizer(self.model)
        self.dm = DataManager()

        # =========================
        # 画板基础尺寸（原始 PIL 大小不变）
        # =========================
        self.base_w, self.base_h = 600, 400
        self.image = Image.new("L", (self.base_w, self.base_h), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None

        # undo 栈
        self.undo_stack = []

        # 画笔参数
        self.brush_size = 20
        self.is_eraser = False

        # =========================
        # 外部卡片容器
        # =========================
        card = QFrame()
        card.setStyleSheet(CARD_STYLE)
        layout = QVBoxLayout(card)

        # =========================
        # 可缩放画布
        # =========================
        self.canvas = QLabel()
        self.canvas.setMinimumSize(300, 200)
        self.canvas.setStyleSheet("background:white; border:1px solid #666;")
        self.canvas.setScaledContents(True)
        layout.addWidget(self.canvas, stretch=1)

        self.update_canvas()

        # =========================
        # 工具按钮区 (Undo/橡皮擦/粗细调节)
        # =========================
        tool_layout = QHBoxLayout()

        self.btn_undo = QPushButton("撤销")
        self.btn_eraser = QPushButton("橡皮：关")
        self.btn_clear = QPushButton("清空")

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setMinimum(5)
        self.brush_slider.setMaximum(40)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.setFixedWidth(150)

        tool_layout.addWidget(self.btn_undo)
        tool_layout.addWidget(self.btn_eraser)
        tool_layout.addWidget(QLabel("粗细："))
        tool_layout.addWidget(self.brush_slider)
        tool_layout.addWidget(self.btn_clear)

        layout.addLayout(tool_layout)

        # 绑定行为
        self.btn_clear.clicked.connect(self.clear)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_eraser.clicked.connect(self.toggle_eraser)
        self.brush_slider.valueChanged.connect(self.update_brush_size)

        # =========================
        # 识别按钮区
        # =========================
        h2 = QHBoxLayout()
        self.btn_recognize = QPushButton("识别数字串")
        self.btn_recognize.setFixedHeight(36)
        h2.addWidget(self.btn_recognize)
        layout.addLayout(h2)

        self.btn_recognize.clicked.connect(self.recognize)

        # =========================
        # 日志输出
        # =========================
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=0)

        self.setLayout(layout)

        info("手写面板（增强版：撤销/橡皮擦/粗细调节/等比例缩放）初始化完成")

    # ---------------------------------------------------------------------
    # 绘图逻辑（含等比例缩放映射）
    # ---------------------------------------------------------------------
    def pil_coords(self, pos):
        """ 将窗口中的鼠标坐标映射回 PIL 画布 """
        cw = self.canvas.width()
        ch = self.canvas.height()

        x_ratio = self.base_w / cw
        y_ratio = self.base_h / ch

        px = int(pos.x() * x_ratio)
        py = int(pos.y() * y_ratio)

        return px, py

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 保存当前状态用于撤销
            self.undo_stack.append(self.image.copy())
            if len(self.undo_stack) > 20:  # 限制栈深度
                self.undo_stack.pop(0)

            self.last_point = event.pos()
            self.draw_point(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.draw_line(self.last_point, event.pos())
            self.last_point = event.pos()

    def draw_point(self, pos):
        px, py = self.pil_coords(pos)
        r = self.brush_size // 2

        color = 255 if self.is_eraser else 0
        self.draw.ellipse((px - r, py - r, px + r, py + r), fill=color)

        self.update_canvas()

    def draw_line(self, p1, p2):
        p1 = self.pil_coords(p1)
        p2 = self.pil_coords(p2)
        color = 255 if self.is_eraser else 0
        self.draw.line([p1, p2], fill=color, width=self.brush_size)
        self.update_canvas()

    # ---------------------------------------------------------------------
    # UI 功能：撤销、橡皮擦、粗细
    # ---------------------------------------------------------------------
    def undo(self):
        if self.undo_stack:
            self.image = self.undo_stack.pop()
            self.draw = ImageDraw.Draw(self.image)
            self.update_canvas()
            self.log.append("撤销一步\n")

    def toggle_eraser(self):
        self.is_eraser = not self.is_eraser
        self.btn_eraser.setText("橡皮：开" if self.is_eraser else "橡皮：关")

    def update_brush_size(self, val):
        self.brush_size = val

    # ---------------------------------------------------------------------
    # 清空画布
    # ---------------------------------------------------------------------
    def clear(self):
        self.image = Image.new("L", (self.base_w, self.base_h), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.update_canvas()
        self.log.append("已清空画布\n")

    # ---------------------------------------------------------------------
    # 显示更新
    # ---------------------------------------------------------------------
    def update_canvas(self):
        img = self.image.convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)

        # scaled 保持自动缩放效果
        self.canvas.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.canvas.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    # ---------------------------------------------------------------------
    # 识别（仍然自动数字串模式）
    # ---------------------------------------------------------------------
    def recognize(self):
        try:
            string, _ = self.dsr.recognize_string(self.image)
            digit_imgs = self.dsr.segment_digits(self.image)

            self.log.append(f"识别结果：{string}\n")

            for img, d in zip(digit_imgs, string):
                ok, label = self.dm.ask_user_confirm(int(d), img)
                if ok:
                    self.dm.save_sample(img, label)

            QMessageBox.information(self, "识别结果", f"数字串：{string}")

        except Exception as e:
            error(f"识别失败：{e}")
            self.log.append(f"识别失败：{e}\n")
