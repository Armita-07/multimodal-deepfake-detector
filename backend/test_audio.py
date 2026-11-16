import argparse
import os
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from src.audio.predict import predict_audio


def gather_files(directory: str, label: int):
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    items = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in exts:
            items.append((os.path.join(directory, fname), label))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    args = parser.parse_args()

    samples = gather_files(args.real_dir, 0) + gather_files(args.fake_dir, 1)
    if not samples:
        print("No audio files found.")
        return

    y_true, y_prob = [], []
    for filepath, label in samples:
        try:
            score = predict_audio(filepath)
            y_true.append(label)
            y_prob.append(score)
            tag = "FAKE" if score >= 0.5 else "REAL"
            print(f"  {tag} ({score:.3f})  {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  ERROR  {os.path.basename(filepath)}: {e}")

    y_true = np.array(y_true)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    print("\n" + "=" * 56)
    print("TEST SET EVALUATION")
    print("=" * 56)
    print(classification_report(y_true, y_pred, target_names=["real", "fake"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
        print(f"\nROC AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
