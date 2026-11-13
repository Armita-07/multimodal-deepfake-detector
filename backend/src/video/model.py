import os
import torch
import timm
from torchvision import transforms

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_WEIGHTS = os.path.join(os.path.dirname(__file__), "video_clf.pth")

_model = None
_transform = None
_FAKE_CLASS_INDEX = 0


def _load():
    global _model, _transform
    if _model is not None:
        return

    if not os.path.isfile(_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found at {_WEIGHTS}")

    _model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    _model.load_state_dict(
        torch.load(_WEIGHTS, map_location=_DEVICE, weights_only=True)
    )
    _model.to(_DEVICE).eval()

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def classify_face(pil_image):
    """
    Evaluates a PIL face image and returns P(Fake) between 0.0 and 1.0.
    """
    _load()
    tensor = _transform(pil_image).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)
    return float(probs[0, _FAKE_CLASS_INDEX].cpu())
