from src.media.validator import validate_size, detect_media_type
from src.image.predict import predict_image
from src.video.predict import predict_video
from src.audio.predict import predict_audio

def route_file(file_path: str) -> dict:
    validate_size(file_path)
    media_type = detect_media_type(file_path)

    if media_type == "image":
        score = predict_image(file_path)
    elif media_type == "video":
        score = predict_video(file_path)
    else:
        score = predict_audio(file_path)

    return {"media_type": media_type, "fake_probability": score}