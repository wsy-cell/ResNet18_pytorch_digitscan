# camera_predict.py
import cv2
import numpy as np
from PIL import Image
import time
import threading
import os
import logging

# 你的识别器接口（需实现 predict_single(PIL->(label, prob))）
from digit_string_recognizer import DigitStringRecognizer
from predict_final import load_model

# 日志系统
from utils.logger import info, error, warn

_default_model = None

def get_default_model():
    global _default_model
    if _default_model is None:
        _default_model = load_model()
    return _default_model

# ---------------------
# 严格数字检测的启发式函数
# ---------------------
def roi_is_likely_digit(gray_patch):
    """
    判断该区域是否为数字
    """
    h, w = gray_patch.shape
    area = w * h
    if area < 120 or area > 10000:
        return False

    _, bw = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.mean(bw == 255)
    if white_ratio < 0.01 or white_ratio > 0.9:
        return False

    ar = w / (h + 1e-8)
    if ar < 0.08 or ar > 6.0:
        return False

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    cnt_area = cv2.contourArea(c)
    bbox_area = w * h
    solidity = cnt_area / (bbox_area + 1e-8)
    if solidity > 0.95:
        return False

    dt = cv2.distanceTransform(255 - bw, cv2.DIST_L2, 3)
    median_width = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0
    if median_width > max(w, h) * 0.25:
        return False

    return True

# ---------------------
# Camera runner 类（线程安全）
# ---------------------
class CameraRunner:
    def __init__(self, model=None, strict=True, skip_frames=1, max_width=640, save_callback=None):
        """
        初始化摄像头识别器
        """
        self.model = model if model is not None else DigitStringRecognizer(get_default_model())
        self.strict = strict
        self.skip_frames = max(1, int(skip_frames))
        self.max_width = max_width
        self.save_callback = save_callback
        self._stop = threading.Event()
        self._thread = None
        self._running = False

        self._frame_times = []
        self.fps = 0.0

    def start(self, cam_index=0):
        if self._running:
            info("摄像头已在运行")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(cam_index,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False
        info("摄像头停止")

    def is_running(self):
        return self._running

    def _run(self, cam_index):
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cap.isOpened():
            error("无法打开摄像头")
            return
        self._running = True
        frame_count = 0
        last_time = time.time()

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                warn("摄像头读取失败")
                break

            # 缩放帧
            h, w = frame.shape[:2]
            if w > self.max_width:
                scale = self.max_width / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                h, w = frame.shape[:2]

            # 跳帧
            frame_count += 1
            if (frame_count % self.skip_frames) != 0:
                now = time.time()
                self._frame_times.append(now - last_time)
                last_time = now
                if len(self._frame_times) > 30:
                    self._frame_times.pop(0)
                self.fps = 1.0 / (np.mean(self._frame_times) + 1e-8) if self._frame_times else 0.0
                cv2.imshow("Camera - Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # 提取 candidate ROIs
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for c in cnts:
                x, y, wc, hc = cv2.boundingRect(c)
                if wc * hc < 200:
                    continue
                pad = 4
                x0 = max(0, x-pad); y0 = max(0, y-pad)
                x1 = min(w-1, x+wc+pad); y1 = min(h-1, y+hc+pad)
                patch = gray[y0:y1, x0:x1]
                candidates.append((x0, y0, x1-x0, y1-y0, patch))

            # 严格模式过滤
            rois = []
            boxes = []
            for (x, y, wc, hc, patch) in candidates:
                ok = True
                if self.strict:
                    ok = roi_is_likely_digit(patch)
                if ok:
                    rois.append(patch)
                    boxes.append((x, y, wc, hc))

            # 进行识别
            preds = []
            for patch in rois:
                pil = Image.fromarray(patch).convert("L")
                p, prob = self.model.predict_single(pil) if hasattr(self.model, "predict_single") else (None, None)
                preds.append((p, prob))

            # 绘制框和标签
            for (x, y, wc, hc), (p, prob) in zip(boxes, preds):
                cv2.rectangle(frame, (x, y), (x+wc, y+hc), (0, 255, 0), 2)
                label_txt = f"{p}" if p is not None else "?"
                cv2.putText(frame, label_txt, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            now = time.time()
            self._frame_times.append(now - last_time)
            last_time = now
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            self.fps = 1.0 / (np.mean(self._frame_times) + 1e-8) if self._frame_times else 0.0
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("Camera - Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and rois:
                if self.save_callback:
                    preds_labels = [p for (p, prob) in preds]
                    self.save_callback(rois, preds_labels)
                else:
                    os.makedirs("./data/camera_feedback", exist_ok=True)
                    idx = len(os.listdir("./data/camera_feedback"))
                    cv2.imwrite(f"./data/camera_feedback/{idx}.png", frame)

        cap.release()
        cv2.destroyAllWindows()
        self._running = False
        info("摄像头线程结束")
