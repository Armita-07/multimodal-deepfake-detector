import os
import mimetypes
import magic
import cv2
from PIL import Image

from src.media.config import (
    MAX_SIZE_BYTES, IMAGE_TYPES, VIDEO_TYPES, AUDIO_TYPES
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def validate_size(file_path: str) -> None:
    """Raises ValueError if the file is empty or too big."""
    size = os.path.getsize(file_path)
    if size == 0:
        raise ValueError("File is empty")
    if size > MAX_SIZE_BYTES:
        raise ValueError("File is too large")


def detect_media_type(file_path: str, original_filename: str = None) -> str:
    """
    Returns 'image', 'video' or 'audio'.
    Uses MIME header detection with verified container parsing fallback for
    formats (like MP4/M4V) that report application/octet-stream on Windows.
    """
    mime = "application/octet-stream"
    try:
        mime = magic.from_file(file_path, mime=True)
    except Exception:
        pass

    if mime in IMAGE_TYPES:
        return "image"
    if mime in VIDEO_TYPES:
        return "video"
    if mime in AUDIO_TYPES:
        return "audio"

    # Fallback: check extension from original filename or temp path
    check_name = original_filename or file_path
    ext = os.path.splitext(check_name)[1].lower()

    # 1. Verify candidate video through OpenCV video decoder
    if ext in VIDEO_EXTS or mime.startswith("video/"):
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frames > 0:
                return "video"

    # 2. Verify candidate image through PIL image decoder
    if ext in IMAGE_EXTS or mime.startswith("image/"):
        try:
            with Image.open(file_path) as img:
                img.verify()
                return "image"
        except Exception:
            pass

    # 3. Verify candidate audio through SoundFile / Librosa
    if ext in AUDIO_EXTS or mime.startswith("audio/"):
        try:
            import soundfile as sf
            sf.info(file_path)
            return "audio"
        except Exception:
            pass

    # 4. Standard mimetypes library check
    guessed_type, _ = mimetypes.guess_type(check_name)
    if guessed_type:
        if guessed_type in IMAGE_TYPES or guessed_type.startswith("image/"):
            return "image"
        if guessed_type in VIDEO_TYPES or guessed_type.startswith("video/"):
            return "video"
        if guessed_type in AUDIO_TYPES or guessed_type.startswith("audio/"):
            return "audio"

    raise ValueError(f"Unsupported file type: {mime}")
