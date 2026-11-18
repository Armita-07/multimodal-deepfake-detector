
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

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np

from PIL import Image, ImageDraw


def generate_image_heatmap(
    image_path,
    predict_fn,
    output_path="reports/image_heatmap.png",
    grid_size=6
):
    """
    Generate an occlusion-sensitivity heatmap.

    predict_fn must accept an image file path or PIL Image and return
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

                try:
                    # In-memory evaluation is much faster
                    masked_score = float(predict_fn(modified))
                except Exception:
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
    heatmap = np.maximum(heatmap, 0.0)

    # Check if any significant manipulation drivers exist:
    # If the image baseline fake probability is negligible (< 0.15) or the maximum drop is noise (< 0.03),
    # there are no manipulation drivers. Do not amplify floating-point noise.
    max_delta = float(heatmap.max())
    significant = (baseline_score >= 0.15) and (max_delta >= 0.03)

    if not significant:
        heatmap = np.zeros((height, width), dtype=np.float32)
        has_artifacts = False
    else:
        has_artifacts = True
        # Expand the patch map to image dimensions.
        heatmap = cv2.resize(
            heatmap,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )
        heatmap = np.clip(heatmap, 0.0, None)
        # Normalize relative to meaningful scale (scale by max drop, bounded by at least 0.15)
        scale_factor = max(max_delta, 0.15)
        heatmap = heatmap / scale_factor

        heatmap = cv2.GaussianBlur(
            heatmap.astype(np.float32),
            (0, 0),
            sigmaX=12
        )
        heatmap = np.clip(heatmap, 0.0, 1.0)

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

    # Dynamic alpha blending: only add color where heatmap intensity > 0.
    # Where heatmap is 0, alpha is 0 (the image is 100% natural, crisp, and untinted).
    alpha = np.expand_dims(heatmap, axis=-1) * 0.65
    overlay = (original_array * (1.0 - alpha) + colored * alpha).astype(np.uint8)

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    Image.fromarray(overlay).save(output_path)

    return {
        "available": True,
        "method": "occlusion_sensitivity",
        "has_artifacts": has_artifacts,
        "max_delta": round(max_delta, 4),
        "baseline_fake_probability": round(
            baseline_score, 4
        ),
        "heatmap_path": output_path,
        "description": (
            "Highlighted clusters indicate spatial regions whose removal reduced the model's fake probability."
            if has_artifacts else
            "Authentic Image: No spatial manipulation clusters detected across the occlusion grid (all regional deltas < 3%)."
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

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0b1120")
    ax.set_facecolor("#0f172a")

    ax.plot(
        frame_numbers,
        scores,
        marker="o",
        markersize=6,
        color="#38bdf8",
        linewidth=2,
        label="Fake Probability"
    )

    ax.axhline(
        y=threshold,
        color="#f43f5e",
        linestyle="--",
        linewidth=1.5,
        label=f"Suspicious Threshold ({threshold})"
    )

    ax.fill_between(
        frame_numbers,
        threshold,
        scores,
        where=scores >= threshold,
        color="#f43f5e",
        alpha=0.25,
        interpolate=True
    )

    ax.set_xlabel("Sampled Frame Number", color="#94a3b8", fontsize=10, fontweight="bold")
    ax.set_ylabel("Fake Probability", color="#94a3b8", fontsize=10, fontweight="bold")
    ax.set_title("Temporal Anomaly Trajectory (FaceForensics++ EfficientNet-B0)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(frame_numbers)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(color="#334155", alpha=0.4, linestyle=":")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", loc="upper left", fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

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


def generate_audio_spectrogram(
    audio_path,
    output_path="reports/audio_spectrogram.png",
    sr=16000,
    max_duration=10
):
    """
    Generate an acoustic waveform and Mel-spectrogram visualization for audio files.
    """
    import librosa

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(waveform) > sr * max_duration:
        waveform = waveform[:sr * max_duration]

    if len(waveform) == 0:
        waveform = np.zeros(sr, dtype=np.float32)

    duration = len(waveform) / sr

    # Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=128,
        fmax=sr // 2
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, facecolor="#0b1120")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    # 1. Waveform Plot
    time_axis = np.linspace(0, duration, num=len(waveform))
    ax1.plot(time_axis, waveform, color="#38bdf8", linewidth=0.9)
    ax1.set_ylabel("Amplitude", color="#94a3b8", fontsize=9, fontweight="bold")
    ax1.set_title(f"Acoustic Speech Waveform ({sr} Hz, Mono)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(color="#334155", alpha=0.3, linestyle=":")

    # 2. Mel-Spectrogram Plot
    img = ax2.imshow(
        mel_spec_db,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, sr // 2],
        cmap="magma"
    )
    ax2.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Frequency (Hz)", color="#94a3b8", fontsize=9, fontweight="bold")
    ax2.set_title("Mel-Frequency Spectrogram (Forensic Acoustic Energy Distribution)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cbar.set_label("Energy (dB)", color="#94a3b8", fontsize=8)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return {
        "available": True,
        "method": "mel_spectrogram_and_waveform",
        "spectrogram_path": output_path,
        "sample_rate": sr,
        "duration_seconds": round(duration, 2),
        "description": "Acoustic waveform and frequency energy distribution across time."
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
