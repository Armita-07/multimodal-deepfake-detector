import os

MAX_SIZE_MB = 100
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-matroska"}
AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "audio/ogg", "audio/x-flac"}

AUDIO_SAMPLE_RATE = 16000   # 16 kHz, what audio models expect
NUM_FRAMES = 16             # frames sampled from each video

BASE_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATIC_DIR = os.path.join(BASE_BACKEND_DIR, "static")
VISUALIZATIONS_DIR = os.path.join(STATIC_DIR, "visualizations")
