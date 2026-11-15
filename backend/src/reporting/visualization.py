
"""
Module 7: Visualization Module

Provides:
1. Image occlusion-sensitivity heatmaps.
2. Video suspicious-frame timelines.

Note:
Image explanations use occlusion sensitivity because
the current CLIP classifier has a custom prediction head.
"""

import os
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw


def generate_image_heatmap(
    image_path,
    predict_fn,
    output_path="reports/image_heatmap.png",
    grid_size=8
):
    """
    Generate an occlusion-sensitivity heatmap.

    predict_fn must accept an image file path and return
    its fake probability between 0 and 1.

    Positive heatmap values indicate regions whose removal
    reduces the fake probability.
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    if grid_size < 2:
        raise ValueError("Grid size must be at least 2.")

    original = Image.open(image_path).convert("RGB")
    width, height = original.size

    baseline_score = float(predict_fn(image_path))

    if not 0 <= baseline_score <= 1:
        raise ValueError("Prediction must be between 0 and 1.")

    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    patch_width = max(1, width // grid_size)
    patch_height = max(1, height // grid_size)

    with tempfile.TemporaryDirectory() as temp_dir:

        for row in range(grid_size):
            for col in range(grid_size):

                x1 = col * patch_width
                y1 = row * patch_height

                x2 = (
                    width if col == grid_size - 1
                    else min(width, x1 + patch_width)
                )

                y2 = (
                    height if row == grid_size - 1
                    else min(height, y1 + patch_height)
                )

                modified = original.copy()
                draw = ImageDraw.Draw(modified)

                # Hide one image patch
                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=(127, 127, 127)
                )

                temp_path = os.path.join(
                    temp_dir, f"patch_{row}_{col}.png"
                )

                modified.save(temp_path)

                masked_score = float(predict_fn(temp_path))

                # How much did hiding this patch reduce
                # the fake probability?
                heatmap[row, col] = (
                    baseline_score - masked_score
                )

    # Keep only regions that support the fake prediction.
    heatmap = np.maximum(heatmap, 0)

    # Expand the patch map to image dimensions.
    heatmap = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(heatmap, 0, None)

    maximum = float(heatmap.max())

    if maximum > 0:
        heatmap = heatmap / maximum

    heatmap = cv2.GaussianBlur(
        heatmap.astype(np.float32),
        (0, 0),
        sigmaX=12
    )

    heatmap = np.clip(heatmap, 0, 1)

    # Convert heatmap into a coloured overlay.
    colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB
    )

    original_array = np.array(original)

    overlay = cv2.addWeighted(
        original_array,
        0.60,
        colored,
        0.40,
        0
    )

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    Image.fromarray(overlay).save(output_path)

    return {
        "available": True,
        "method": "occlusion_sensitivity",
        "baseline_fake_probability": round(
            baseline_score, 4
        ),
        "heatmap_path": output_path,
        "description": (
            "Highlighted regions indicate areas where "
            "occlusion reduced the model's fake probability. "
            "This is a model-sensitivity explanation, "
            "not proof of manipulation."
        )
    }


def generate_video_timeline(
    frame_scores,
    output_path="reports/video_timeline.png",
    threshold=0.5
):
    """
    Create a timeline of fake probabilities for video frames.

    frame_scores: list of probabilities between 0 and 1.
    """

    if not frame_scores:
        raise ValueError("Frame scores cannot be empty.")

    scores = np.asarray(frame_scores, dtype=float)

    if not np.all(np.isfinite(scores)):
        raise ValueError("Frame scores must be finite.")

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError(
            "Frame scores must be between 0 and 1."
        )

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    frame_numbers = np.arange(1, len(scores) + 1)

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    plt.figure(figsize=(10, 4))

    plt.plot(
        frame_numbers,
        scores,
        marker="o",
        color="blue",
        label="Fake probability"
    )

    plt.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        label="Suspicious threshold"
    )

    plt.fill_between(
        frame_numbers,
        threshold,
        scores,
        where=scores >= threshold,
        color="red",
        alpha=0.20
    )

    plt.xlabel("Frame number")
    plt.ylabel("Fake probability")
    plt.title("Video Deepfake Detection Timeline")

    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    suspicious_frames = (
        frame_numbers[scores >= threshold].tolist()
    )

    return {
        "available": True,
        "method": "frame_score_timeline",
        "timeline_path": output_path,
        "total_frames": len(scores),
        "suspicious_frames": suspicious_frames,
        "threshold": threshold,
        "average_fake_probability": round(
            float(np.mean(scores)), 4
        )
    }


if __name__ == "__main__":

    # Example frame scores for testing the timeline.
    example_scores = [
        0.12, 0.18, 0.25, 0.72,
        0.83, 0.91, 0.68, 0.30,
        0.20, 0.78, 0.85, 0.15,
        0.22, 0.35, 0.80, 0.88
    ]

    result = generate_video_timeline(example_scores)

    print("Video timeline generated!")
    print(result)
