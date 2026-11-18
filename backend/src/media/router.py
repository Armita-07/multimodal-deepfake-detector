
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
from datetime import datetime, timezone

from src.media.config import VISUALIZATIONS_DIR
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
from src.reporting.visualization import (
    generate_image_heatmap,
    generate_video_timeline,
    generate_audio_spectrogram
)


def route_file(
    file_path: str,
    original_filename: str = None
) -> dict:
    """
    Analyze an uploaded media file and generate
    a prediction using the appropriate detection model,
    accompanied by visual explanations and structured metrics.
    """

    # Step 1: Validate uploaded file size
    validate_size(file_path)

    # Step 2: Detect media type
    media_type = detect_media_type(
        file_path,
        original_filename=original_filename
    )

    visual_explanation = {"available": False}
    frame_scores = None
    timestamp_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    # Step 3: Run the appropriate detection model & generate visualizations
    if media_type == "image":

        score = predict_image(file_path)

        fusion_result = fuse_predictions(
            image_score=score
        )

        viz_filename = f"{timestamp_id}_heatmap.png"
        viz_path = os.path.join(VISUALIZATIONS_DIR, viz_filename)

        try:
            viz_data = generate_image_heatmap(
                image_path=file_path,
                predict_fn=predict_image,
                output_path=viz_path,
                grid_size=6
            )
            visual_explanation = {
                "available": True,
                "type": "heatmap",
                "url": f"/static/visualizations/{viz_filename}",
                "method": viz_data.get("method", "occlusion_sensitivity"),
                "has_artifacts": viz_data.get("has_artifacts"),
                "max_delta": viz_data.get("max_delta"),
                "baseline_fake_probability": viz_data.get("baseline_fake_probability"),
                "description": viz_data.get("description")
            }
        except Exception as e:
            visual_explanation = {
                "available": False,
                "error": str(e),
                "message": "Heatmap generation failed."
            }

    elif media_type == "video":

        score, frame_scores = predict_video(
            file_path,
            return_frame_scores=True
        )

        fusion_result = fuse_predictions(
            video_score=score
        )

        viz_filename = f"{timestamp_id}_timeline.png"
        viz_path = os.path.join(VISUALIZATIONS_DIR, viz_filename)

        has_faces = bool(frame_scores and len(frame_scores) > 0)
        timeline_scores = frame_scores if has_faces else [0.5] * 16

        try:
            viz_data = generate_video_timeline(
                frame_scores=timeline_scores,
                output_path=viz_path,
                threshold=0.5
            )
            visual_explanation = {
                "available": True,
                "type": "timeline",
                "url": f"/static/visualizations/{viz_filename}",
                "method": viz_data.get("method", "frame_score_timeline"),
                "has_faces_detected": has_faces,
                "total_frames": viz_data.get("total_frames"),
                "suspicious_frames": viz_data.get("suspicious_frames") if has_faces else [],
                "threshold": viz_data.get("threshold"),
                "average_fake_probability": viz_data.get("average_fake_probability"),
                "description": (
                    "Frame-by-frame deepfake probability timeline across sampled video frames."
                    if has_faces else
                    "No human faces detected across video samples. Neutral baseline trajectory."
                )
            }
        except Exception as e:
            visual_explanation = {
                "available": False,
                "error": str(e),
                "message": "Video timeline generation failed."
            }

    elif media_type == "audio":

        score = predict_audio(file_path)

        fusion_result = fuse_predictions(
            audio_score=score
        )

        viz_filename = f"{timestamp_id}_spectrogram.png"
        viz_path = os.path.join(VISUALIZATIONS_DIR, viz_filename)

        try:
            viz_data = generate_audio_spectrogram(
                audio_path=file_path,
                output_path=viz_path,
                sr=16000,
                max_duration=10
            )
            visual_explanation = {
                "available": True,
                "type": "spectrogram",
                "url": f"/static/visualizations/{viz_filename}",
                "method": viz_data.get("method", "mel_spectrogram_and_waveform"),
                "sample_rate": viz_data.get("sample_rate"),
                "duration_seconds": viz_data.get("duration_seconds"),
                "description": viz_data.get("description")
            }
        except Exception as e:
            visual_explanation = {
                "available": False,
                "error": str(e),
                "message": "Audio spectrogram generation failed."
            }

    else:
        raise ValueError("Unsupported media type.")

    # Step 4: Generate structured report with visual explanation
    report = generate_report(
        fusion_result=fusion_result,
        filename=(
            original_filename
            or os.path.basename(file_path)
        ),
        media_type=media_type,
        visual_explanation=visual_explanation
    )

    # Step 5: Save JSON report
    report_path = save_report(report)

    # Step 6: Assemble metrics breakdown
    metrics_data = {
        "decision_threshold": 0.5,
        "raw_score": round(float(score), 4),
        "fake_probability": round(float(fusion_result["fake_probability"]), 4),
        "real_probability": round(float(fusion_result["real_probability"]), 4),
        "confidence": round(float(fusion_result["confidence"]), 4)
    }

    if media_type == "video":
        has_faces = bool(frame_scores and len(frame_scores) > 0)
        suspicious_list = [i + 1 for i, s in enumerate(frame_scores) if s >= 0.5] if has_faces else []
        metrics_data["video_frame_metrics"] = {
            "has_faces_detected": has_faces,
            "total_frames_analyzed": len(frame_scores) if has_faces else 16,
            "suspicious_frames_count": len(suspicious_list),
            "suspicious_frame_indices": suspicious_list,
            "frame_scores": [round(float(s), 4) for s in frame_scores] if has_faces else [0.5] * 16
        }

    # Step 7: Return results to FastAPI and frontend
    heatmap_url = visual_explanation.get("url") if visual_explanation.get("available") else None

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

        "demo": False,

        "visualization": visual_explanation,

        "heatmap_url": heatmap_url,

        "metrics": metrics_data
    }
