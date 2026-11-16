import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MARGIN_RATIO = 0.3
_FACE_SIZE = 224

_mtcnn = None


def _load():
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(
            keep_all=False,
            post_process=False,
            select_largest=True,
            device=_DEVICE,
        )


def detect_and_crop(frame_bgr):
    """
    Detects the primary face in a BGR frame and returns a 224x224 PIL image,
    or None if no face is detected.
    """
    _load()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, probs = _mtcnn.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return None

    best = int(np.argmax(probs))
    x1, y1, x2, y2 = boxes[best]

    w, h = x2 - x1, y2 - y1
    margin = _MARGIN_RATIO * max(w, h)
    img_h, img_w = frame_bgr.shape[:2]
    x1 = max(0, int(x1 - margin))
    y1 = max(0, int(y1 - margin))
    x2 = min(img_w, int(x2 + margin))
    y2 = min(img_h, int(y2 + margin))

    if x2 <= x1 or y2 <= y1:
        return None

    return pil_img.crop((x1, y1, x2, y2)).resize(
        (_FACE_SIZE, _FACE_SIZE), Image.BILINEAR
    )
