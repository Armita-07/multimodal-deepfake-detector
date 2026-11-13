import os
import magic
from src.media.config import (
    MAX_SIZE_BYTES, IMAGE_TYPES, VIDEO_TYPES, AUDIO_TYPES
)

def validate_size(file_path: str) -> None:
    """Raises ValueError if the file is empty or too big."""
    size = os.path.getsize(file_path)
    if size == 0:
        raise ValueError("File is empty")
    if size > MAX_SIZE_BYTES:
        raise ValueError("File is too large")

def detect_media_type(file_path: str) -> str:
    """Returns 'image', 'video' or 'audio'. Raises ValueError if unsupported."""
    mime = magic.from_file(file_path, mime=True)   # reads actual content
    if mime in IMAGE_TYPES:
        return "image"
    if mime in VIDEO_TYPES:
        return "video"
    if mime in AUDIO_TYPES:
        return "audio"
    raise ValueError(f"Unsupported file type: {mime}")