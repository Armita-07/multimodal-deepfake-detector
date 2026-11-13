import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_WEIGHTS = os.path.join(os.path.dirname(__file__), "audio_clf.pth")

_model = None
_FAKE_CLASS_INDEX = 1


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            output_hidden_states=False,
        )
        self.backbone.freeze_feature_encoder()
        self.head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.backbone(waveform).last_hidden_state
        pooled = features.mean(dim=1)
        return self.head(pooled)


def _load():
    global _model
    if _model is not None:
        return

    if not os.path.isfile(_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found at {_WEIGHTS}")

    _model = AudioClassifier()
    state = torch.load(_WEIGHTS, map_location=_DEVICE, weights_only=False)
    _model.load_state_dict(state)
    _model.to(_DEVICE).eval()


def classify_audio(waveform_np):
    _load()
    tensor = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)
    return float(probs[0, _FAKE_CLASS_INDEX].cpu())
