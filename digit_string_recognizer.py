"""
DigitStringRecognizer
- 更稳健的分割（自适应阈值 + 边缘清洗 + 面积/比例过滤）
- 提供 predict_single(img_pil) 供摄像头/GUI 单位识别调用（返回 int, prob）
- 提供 segment_digits(img_pil) 返回分割后的 PIL 图像列表（按左到右）
- 提供 recognize_string(img_pil) 返回 (string, probs)
依赖：predict_final.load_model/predict, preprocess.preprocess_custom_image
"""

import cv2
import numpy as np
from PIL import Image
from core.preprocess import preprocess_custom_image
from core.predict_final import predict, load_model
import warnings

warnings.filterwarnings("ignore")

class DigitStringRecognizer:
    def __init__(self, model=None):
        """
        model: 如果传 None，会在需要时自动通过 load_model() 加载
        """
        self.model = model

    def ensure_model(self):
        if self.model is None:
            self.model = load_model()
        return self.model

    def preprocess_roi(self, roi_np):
        """
        roi_np: numpy 灰度图 (H,W) or PIL
        返回：PIL 28x28（MNIST 风格）
        """
        if isinstance(roi_np, np.ndarray):
            img = Image.fromarray(roi_np)
        else:
            img = roi_np
        return preprocess_custom_image(img)

    def segment_digits(self, img_pil, debug=False):
        """
        输入：PIL 图像（整串手写/摄像头帧）
        输出：[(x, PIL_img_roi), ...] 按 x 排序（左到右）
        算法要点：
          - 转灰度、自适应阈值
          - 开闭运算去噪
          - 轮廓过滤（面积、宽高比、像素密度）
        """
        img = np.array(img_pil.convert("L"))

        # 自适应阈值（比固定阈值更稳）
        th = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 8)

        # 形态学处理：去噪并连通微小断裂（闭运算）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        H, W = th.shape

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w*h
            # 过滤太小
            if area < 400:
                continue
            # 宽高比过滤：数字通常在 0.15 ~ 6 之间
            ar = w / (h + 1e-8)
            if ar < 0.12 or ar > 6:
                continue
            # 像素密度（白色像素比例）
            patch = th[y:y+h, x:x+w]
            density = np.mean(patch == 255)
            if density < 0.03:  # 太稀疏可能是噪声
                continue
            candidates.append((x, y, w, h))

        if not candidates:
            return []

        # 合并重叠或近邻小框（防止数字被分成两半）
        candidates = sorted(candidates, key=lambda b: b[0])
        merged = []
        cur = list(candidates[0])
        for x,y,w,h in candidates[1:]:
            if x <= cur[0] + cur[2] + 8:  # 如果重叠或邻近（阈值8）
                # 合并
                x0 = min(cur[0], x)
                y0 = min(cur[1], y)
                x1 = max(cur[0]+cur[2], x+w)
                y1 = max(cur[1]+cur[3], y+h)
                cur = [x0, y0, x1-x0, y1-y0]
            else:
                merged.append(tuple(cur))
                cur = [x,y,w,h]
        merged.append(tuple(cur))

        # 转为 PIL 截图并排序
        rois = []
        for (x,y,w,h) in merged:
            crop = img_pil.crop((x, y, x+w, y+h))
            rois.append((x, crop))

        rois = sorted(rois, key=lambda t: t[0])
        return rois

    def predict_single(self, img_pil):
        """
        直接对单个 PIL 图像做预测（内部做 preprocess_custom_image）
        返回：pred(int), prob(list of 10 floats)
        """
        model = self.ensure_model()
        x = self.preprocess_roi(img_pil)
        pred, prob = predict(model, x)  # predict 已返回 (pred, prob)
        return pred, prob

    def recognize_string(self, img_pil):
        """
        识别整串数字（返回字符串 + 每位概率）
        """
        rois = self.segment_digits(img_pil)
        if not rois:
            return "", []

        results = []
        probs = []
        for x, roi in rois:
            pred, prob = self.predict_single(roi)
            results.append(str(pred))
            probs.append(prob)
        return "".join(results), probs


# 如果做为脚本测试
if __name__ == "__main__":
    # 简易测试（需要你自行指定图片路径）
    m = load_model()
    dsr = DigitStringRecognizer(m)
    from PIL import Image
    im = Image.open("test_line.png") if False else None
    if im:
        s, p = dsr.recognize_string(im)
        print(s, p)
