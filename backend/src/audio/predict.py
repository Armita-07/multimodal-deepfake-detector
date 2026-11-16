import librosa
from src.audio.model import classify_audio

TARGET_SR = 16000
MAX_DURATION = 10


def predict_audio(path: str) -> float:
    waveform, _ = librosa.load(path, sr=TARGET_SR, mono=True)

    max_samples = TARGET_SR * MAX_DURATION
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    if len(waveform) == 0:
        return 0.5

    return classify_audio(waveform)
