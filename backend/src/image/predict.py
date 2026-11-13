import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_NAME = "openai/clip-vit-base-patch32"
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "image_clf_combined.npz")

_clip = None
_processor = None
_w = None
_b = None


def _load():
    """Loads CLIP and the trained weights once, the first time they are needed."""
    global _clip, _processor, _w, _b
    if _clip is None:
        _clip = CLIPModel.from_pretrained(_MODEL_NAME).to(_DEVICE).eval()
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        data = np.load(_WEIGHTS_PATH)
        _w = data["w"]
        _b = data["b"]


def _embed(pixel_values):
    """Image tensor -> 512-number fingerprint (same steps as in training)."""
    out = _clip.vision_model(pixel_values=pixel_values)
    f = _clip.visual_projection(out.pooler_output)
    return f / f.norm(dim=-1, keepdim=True)


def predict_image(path: str) -> float:
    """Returns the probability (0 to 1) that the image is fake."""
    _load()
    img = Image.open(path).convert("RGB")
    inputs = _processor(images=img, return_tensors="pt")
    with torch.no_grad():
        f = _embed(inputs["pixel_values"].to(_DEVICE))
    z = float(f.cpu().numpy()[0] @ _w + _b[0])
    return float(1 / (1 + np.exp(-z)))