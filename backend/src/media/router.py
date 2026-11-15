
"""
Media Router

Routes uploaded images, videos and audio files
to their respective detection models.

Connects:
1. Image Detection
2. Video Detection
3. Audio Detection
4. Module 6 - Multimodal Fusion
5. Module 7 - Reporting
"""

import os

from src.media.validator import (
    validate_size,
    detect_media_type
)

from src.image.predict import predict_image
from src.video.predict import predict_video
from src.audio.predict import predict_audio

from src.fusion.fusion import fuse_predictions

from src.reporting.report import (
    generate_report,
    save_report
)


def route_file(
    file_path: str,
    original_filename: str = None
) -> dict:
    """
    Analyze an uploaded media file and generate
    a prediction using the appropriate detection model.
    """

    # Step 1: Validate uploaded file size
    validate_size(file_path)

    # Step 2: Detect media type
    media_type = detect_media_type(file_path)

    # Step 3: Run the appropriate detection model
    if media_type == "image":

        score = predict_image(file_path)

        fusion_result = fuse_predictions(
            image_score=score
        )

    elif media_type == "video":

        score = predict_video(file_path)

        fusion_result = fuse_predictions(
            video_score=score
        )

    elif media_type == "audio":

        score = predict_audio(file_path)

        fusion_result = fuse_predictions(
            audio_score=score
        )

    else:
        raise ValueError("Unsupported media type.")

    # Step 4: Generate structured report
    report = generate_report(
        fusion_result=fusion_result,
        filename=(
            original_filename
            or os.path.basename(file_path)
        ),
        media_type=media_type
    )

    # Step 5: Save JSON report
    report_path = save_report(report)

    # Step 6: Return results to FastAPI and frontend
    return {
        "label": fusion_result["predicted_label"],

        "fake_probability": fusion_result["fake_probability"],

        "real_probability": fusion_result["real_probability"],

        "confidence": fusion_result["confidence"],

        "modality": media_type,

        "modalities_used": fusion_result["modalities_used"],

        "fusion_method": fusion_result["fusion_method"],

        "explanation": (
            f"{media_type.capitalize()} analysis completed. "
            f"Detection result: "
            f"{fusion_result['predicted_label']}."
        ),

        "report_path": report_path,

        "demo": False
    }
