import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

TARGET_SR = 16000
MAX_SAMPLES = TARGET_SR * 4


class ASVspoofDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.samples = []
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(root_dir, split, label_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, fname), label_id))
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)

        if len(wav) > MAX_SAMPLES:
            start = np.random.randint(0, len(wav) - MAX_SAMPLES)
            wav = wav[start : start + MAX_SAMPLES]
        else:
            wav = np.pad(wav, (0, max(0, MAX_SAMPLES - len(wav))))

        return torch.tensor(wav, dtype=torch.float32), label


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            output_hidden_states=False,
        )
        self.backbone.freeze_feature_encoder()
        self.head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, waveform):
        features = self.backbone(waveform).last_hidden_state
        pooled = features.mean(dim=1)
        return self.head(pooled)


def train():
    data_root = "/content/asvspoof_ready"
    epochs = 5
    batch_size = 8
    lr = 1e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = ASVspoofDataset(data_root, "train")
    dev_ds = ASVspoofDataset(data_root, "dev")
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AudioClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_wav, batch_label in train_loader:
            batch_wav = batch_wav.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            logits = model(batch_wav)
            loss = criterion(logits, batch_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_wav, batch_label in dev_loader:
                batch_wav = batch_wav.to(device)
                batch_label = batch_label.to(device)
                logits = model(batch_wav)
                val_loss += criterion(logits, batch_label).item()
                preds = logits.argmax(dim=1)
                correct += (preds == batch_label).sum().item()
                total += batch_label.size(0)

        avg_val_loss = val_loss / len(dev_loader)
        val_acc = correct / total
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for batch_wav, batch_label in dev_loader:
            batch_wav = batch_wav.to(device)
            logits = model(batch_wav)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_true.extend(batch_label.numpy())
            all_prob.extend(probs.cpu().numpy())

    all_true = np.array(all_true)
    all_pred = (np.array(all_prob) >= 0.5).astype(int)

    print("=" * 56)
    print("TEST SET EVALUATION")
    print("=" * 56)
    print(classification_report(all_true, all_pred, target_names=["real", "fake"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_pred))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    cm = confusion_matrix(all_true, all_pred)
    axes[0, 0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=16)
    axes[0, 0].set_xticks([0, 1]); axes[0, 0].set_xticklabels(["Real", "Fake"])
    axes[0, 0].set_yticks([0, 1]); axes[0, 0].set_yticklabels(["Real", "Fake"])
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("Actual")
    axes[0, 0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(all_true, all_prob)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, "b-", label=f"AUC = {roc_auc:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--")
    axes[0, 1].set_xlabel("False Positive Rate"); axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve"); axes[0, 1].legend()

    real_scores = np.array(all_prob)[all_true == 0]
    fake_scores = np.array(all_prob)[all_true == 1]
    axes[1, 0].hist(real_scores, bins=30, alpha=0.6, label="Real", color="green")
    axes[1, 0].hist(fake_scores, bins=30, alpha=0.6, label="Fake", color="red")
    axes[1, 0].axvline(0.5, color="black", linestyle="--", label="Threshold")
    axes[1, 0].set_xlabel("Fake Probability"); axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Score Distribution"); axes[1, 0].legend()

    epochs_range = range(1, epochs + 1)
    axes[1, 1].plot(epochs_range, history["train_loss"], "b-o", label="Train Loss")
    axes[1, 1].plot(epochs_range, history["val_loss"], "r-o", label="Val Loss")
    ax2 = axes[1, 1].twinx()
    ax2.plot(epochs_range, history["val_acc"], "g-s", label="Val Acc")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    axes[1, 1].set_title("Training Curves")
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/audio_eval_plots.png", dpi=150)
    plt.show()

    torch.save(model.state_dict(), "/content/drive/MyDrive/audio_clf.pth")
    file_size = os.path.getsize("/content/drive/MyDrive/audio_clf.pth") / (1024 * 1024)
    print(f"Saved: /content/drive/MyDrive/audio_clf.pth ({file_size:.1f} MB)")


if __name__ == "__main__":
    train()
