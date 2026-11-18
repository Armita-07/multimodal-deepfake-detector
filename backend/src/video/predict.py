import numpy as np
from src.media.extractor import extract_frames
from src.video.face_detector import detect_and_crop
from src.video.model import classify_face


def predict_video(path: str, return_frame_scores: bool = False):
    """
    Predicts the probability (0.0 to 1.0) that a video is a deepfake.
    Samples 16 frames, crops faces via MTCNN, classifies with EfficientNet-B0,
    and returns the mean score across detected faces.
    If return_frame_scores is True, returns a tuple: (mean_score, frame_scores).
    """
    frames = extract_frames(path)
    if not frames:
        return (0.5, []) if return_frame_scores else 0.5

    scores = []
    for frame in frames:
        face_crop = detect_and_crop(frame)
        if face_crop is None:
            continue
        scores.append(float(classify_face(face_crop)))

    if not scores:
        return (0.5, []) if return_frame_scores else 0.5

    mean_score = float(np.mean(scores))
    if return_frame_scores:
        return mean_score, scores
    return mean_score
