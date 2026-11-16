
"""
Module 7: Reporting Module

Creates structured detection reports from
the multimodal fusion results.
"""

import json
from pathlib import Path
from datetime import datetime, timezone


DISCLAIMER = (
    "This result is an AI-generated estimate, not proof "
    "of authenticity or manipulation. False positives and "
    "false negatives are possible. Human verification "
    "is recommended."
)


def generate_report(
    fusion_result: dict,
    filename: str = "uploaded_media",
    media_type: str = "unknown",
    visual_explanation: dict = None
) -> dict:
    """
    Generate a structured report using Module 6 output.
    """

    if not isinstance(fusion_result, dict):
        raise ValueError("Fusion result must be a dictionary.")

    required_fields = [
        "predicted_label",
        "fake_probability",
        "real_probability",
        "confidence",
        "modalities_used"
    ]

    for field in required_fields:
        if field not in fusion_result:
            raise ValueError(
                f"Missing fusion result field: {field}"
            )

    label = fusion_result["predicted_label"]

    if label not in ("REAL", "FAKE"):
        raise ValueError("Invalid prediction label.")

    fake_probability = float(
        fusion_result["fake_probability"]
    )

    real_probability = float(
        fusion_result["real_probability"]
    )

    confidence = float(fusion_result["confidence"])

    for value in (
        fake_probability,
        real_probability,
        confidence
    ):
        if not 0 <= value <= 1:
            raise ValueError(
                "Probabilities and confidence must be between 0 and 1."
            )

    report = {
        "report_id": datetime.now(
            timezone.utc
        ).strftime("%Y%m%d%H%M%S%f"),

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "filename": filename,

        "media_type": media_type,

        "prediction": {
            "label": label,
            "fake_probability": round(fake_probability, 4),
            "real_probability": round(real_probability, 4),
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2)
        },

        "analysis": {
            "modalities_used": fusion_result["modalities_used"],
            "fusion_method": fusion_result.get(
                "fusion_method",
                "unknown"
            )
        },

        "visual_explanation": visual_explanation or {
            "available": False,
            "message": "No visual explanation generated yet."
        },

        "disclaimer": DISCLAIMER
    }

    return report


def save_report(
    report: dict,
    output_directory: str = "reports"
) -> str:
    """
    Save the generated report as a JSON file.
    """

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    report_id = report.get("report_id", "report")

    file_path = output_path / f"{report_id}.json"

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return str(file_path)


if __name__ == "__main__":

    # Example fusion result from Module 6
    sample_result = {
        "predicted_label": "FAKE",
        "fake_probability": 0.79,
        "real_probability": 0.21,
        "confidence": 0.79,
        "modalities_used": [
            "image",
            "video",
            "audio"
        ],
        "fusion_method": "weighted_late_fusion"
    }

    # Generate report
    report = generate_report(
        fusion_result=sample_result,
        filename="sample_video.mp4",
        media_type="video"
    )

    # Display report
    print(json.dumps(report, indent=4))

    # Save report
    saved_path = save_report(report)

    print("\nReport saved successfully!")
    print("Location:", saved_path)
