import os
import subprocess
import cv2
import numpy as np
from src.media.config import AUDIO_SAMPLE_RATE, NUM_FRAMES

def extract_audio(video_path: str, out_path: str = None) -> str:
    """Pulls the audio track out of a video as a 16 kHz mono WAV file."""
    if out_path is None:
        out_path = os.path.splitext(video_path)[0] + "_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                          # no video
        "-ac", "1",                     # mono
        "-ar", str(AUDIO_SAMPLE_RATE),  # 16 kHz
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path

def extract_frames(video_path: str, num_frames: int = NUM_FRAMES) -> list:
    """Returns `num_frames` frames spread evenly across the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indexes = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for i in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames