
"""
Module 6: Multimodal Fusion Module

Combines image, video and audio fake probabilities
using weighted late fusion.

Each input probability must be between 0 and 1.
Missing modalities are automatically ignored.
"""

from typing import Optional
import math


# Default weights for each expert
DEFAULT_WEIGHTS = {
    "image": 0.30,
    "video": 0.40,
    "audio": 0.30
}


def validate_score(score: Optional[float], name: str):
    """Validate a probability score."""

    if score is None:
        return None

    if isinstance(score, bool):
        raise ValueError(f"{name} score must be a number.")

    score = float(score)

    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(
            f"{name} score must be between 0 and 1."
        )

    return score


def fuse_predictions(
    image_score: Optional[float] = None,
    video_score: Optional[float] = None,
    audio_score: Optional[float] = None,
    weights: Optional[dict] = None,
    threshold: float = 0.5
) -> dict:
    """
    Combine available modality scores.

    Returns the final fake probability, predicted label,
    confidence and modalities used.
    """

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    scores = {
        "image": validate_score(image_score, "Image"),
        "video": validate_score(video_score, "Video"),
        "audio": validate_score(audio_score, "Audio")
    }

    active_scores = {
        name: score
        for name, score in scores.items()
        if score is not None
    }

    # No available modality
    if not active_scores:
        raise ValueError(
            "At least one modality score is required."
        )

    selected_weights = weights or DEFAULT_WEIGHTS

    # Check weights
    active_weights = {}

    for name in active_scores:
        weight = float(selected_weights.get(name, 0))

        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"Invalid weight for {name}."
            )

        active_weights[name] = weight

    total_weight = sum(active_weights.values())

    if total_weight <= 0:
        raise ValueError(
            "At least one available modality must have "
            "a positive weight."
        )

    # Weighted late fusion
    fake_probability = sum(
        active_scores[name] * active_weights[name]
        for name in active_scores
    ) / total_weight

    fake_probability = max(0.0, min(1.0, fake_probability))

    # Final classification
    predicted_label = (
        "FAKE"
        if fake_probability >= threshold
        else "REAL"
    )

    # Confidence in the predicted class
    confidence = (
        fake_probability
        if predicted_label == "FAKE"
        else 1.0 - fake_probability
    )

    return {
        "predicted_label": predicted_label,
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(1.0 - fake_probability, 4),
        "confidence": round(confidence, 4),
        "modalities_used": list(active_scores.keys()),
        "fusion_method": "weighted_late_fusion"
    }


if __name__ == "__main__":

    # Example 1: All three modalities available
    result = fuse_predictions(
        image_score=0.80,
        video_score=0.70,
        audio_score=0.90
    )

    print("All modalities:")
    print(result)

    # Example 2: Image only
    image_only = fuse_predictions(
        image_score=0.85
    )

    print("\nImage only:")
    print(image_only)

    # Example 3: Image and audio only
    partial = fuse_predictions(
        image_score=0.80,
        audio_score=0.60
    )

    print("\nPartial modalities:")
    print(partial)
