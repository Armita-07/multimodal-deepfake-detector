import os
import sys
import time
import numpy as np

from src.video.predict import predict_video

TEST_DIR = sys.argv[1] if len(sys.argv) > 1 else "test_videos"
EXT = (".mp4", ".avi", ".webm", ".mkv", ".mov")


def main():
    if not os.path.isdir(TEST_DIR):
        print(f"Directory not found: {TEST_DIR}")
        sys.exit(1)

    labels = []
    probabilities = []
    predictions = []
    timings = []

    print(f"Evaluating videos in: {os.path.abspath(TEST_DIR)}")
    print("─" * 70)

    for folder, _, files in os.walk(TEST_DIR):
        for name in sorted(files):
            if not name.lower().endswith(EXT):
                continue

            path = os.path.join(folder, name)
            sub = os.path.relpath(folder, TEST_DIR).lower()

            if "fake" in sub:
                truth = 1
            elif "real" in sub:
                truth = 0
            else:
                continue

            t0 = time.time()
            score = predict_video(path)
            elapsed = time.time() - t0

            guess = 1 if score >= 0.5 else 0
            status = "✓" if guess == truth else "✗"

            labels.append(truth)
            predictions.append(guess)
            probabilities.append(score)
            timings.append(elapsed)

            truth_str = "FAKE" if truth else "REAL"
            guess_str = "FAKE" if guess else "REAL"
            print(
                f"  {status} {name:35s} truth={truth_str:4s}  "
                f"score={score:.3f}  guess={guess_str:4s}  ({elapsed:.1f}s)"
            )

    if not labels:
        print("No labeled test videos found.")
        return

    labels = np.array(labels)
    predictions = np.array(predictions)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 55)
    print("VIDEO EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Total Videos  : {len(labels)}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"Avg Time/Video: {np.mean(timings):.2f}s")


if __name__ == "__main__":
    main()
