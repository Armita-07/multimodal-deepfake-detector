# ---
# Module 4 — Video Deepfake Detection: Training Notebook
# Run this on Google Colab with GPU runtime (Runtime → Change runtime type → T4 GPU)
# ---

# %% [markdown]
# # Module 4: Train EfficientNet-B0 for Video Deepfake Detection
#
# This notebook trains a binary classifier (real vs fake) on face crops
# extracted from FaceForensics++ videos.
#
# **Before running:** Upload your `dataset/` folder to Google Drive with this structure:
# ```
# My Drive/
#   deepfake_dataset/
#     train/real/  train/fake/
#     val/real/    val/fake/
#     test/real/   test/fake/
# ```

# %% — Cell 1: Install dependencies
# !pip install timm facenet-pytorch -q

# %% — Cell 2: Imports and setup
import os
import time
import copy
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

# Mount Google Drive
from google.colab import drive
drive.mount("/content/drive")

# ── CONFIGURE THIS PATH ──
DATASET_ROOT = "/content/drive/MyDrive/deepfake_dataset"
# ──────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% — Cell 3: Data loading with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05)
    ], p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_set = datasets.ImageFolder(f"{DATASET_ROOT}/train", train_transform)
val_set   = datasets.ImageFolder(f"{DATASET_ROOT}/val",   eval_transform)
test_set  = datasets.ImageFolder(f"{DATASET_ROOT}/test",  eval_transform)

print(f"Classes: {train_set.classes}")
print(f"Class-to-index: {train_set.class_to_idx}")
print(f"Train: {len(train_set):,}  |  Val: {len(val_set):,}  |  Test: {len(test_set):,}")

BATCH_SIZE = 64

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# %% — Cell 4: Visualize a batch of training samples
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
images, labels = next(iter(train_loader))
for i, ax in enumerate(axes.flat):
    if i >= len(images):
        break
    img = images[i].permute(1, 2, 0).numpy()
    img = np.clip(img * std + mean, 0, 1)
    ax.imshow(img)
    ax.set_title(train_set.classes[labels[i]], fontsize=9)
    ax.axis("off")
plt.suptitle("Training Samples (augmented)", fontsize=13)
plt.tight_layout()
plt.savefig("sample_batch.png", dpi=100)
plt.show()

# %% — Cell 5: Build the model
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:>10,}")
print(f"Trainable parameters: {trainable_params:>10,}")

# %% — Cell 6: Training setup
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [],  "val_acc": [],
}
best_val_acc = 0.0
best_model_state = None


def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total

# %% — Cell 7: Train!
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
      f"{'Val Loss':>10}  {'Val Acc':>9}  {'Time':>5}")
print("─" * 60)

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                       optimizer, train=True)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, train=False)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - t0
    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        marker = " ← best"

    print(f"{epoch+1:>5d}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
          f"{val_loss:>10.4f}  {val_acc:>9.4f}  {elapsed:>4.0f}s{marker}")

print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# %% — Cell 8: Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs_range = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs_range, history["train_loss"], "o-", label="Train")
ax1.plot(epochs_range, history["val_loss"], "o-", label="Validation")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history["train_acc"], "o-", label="Train")
ax2.plot(epochs_range, history["val_acc"], "o-", label="Validation")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Training & Validation Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Cell 9: Evaluate on test set
model.load_state_dict(best_model_state)
model.eval()

all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        # class 0 = fake (alphabetical), so probs[:, 0] = P(fake)
        all_probs.extend(probs[:, 0].cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

print("=" * 60)
print("FRAME-LEVEL TEST SET RESULTS")
print("=" * 60)
print()
print(classification_report(
    all_labels, all_preds,
    target_names=test_set.classes
))

try:
    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC: {auc:.4f}")
except ValueError:
    print("ROC-AUC: could not compute (single class)")

# %% — Cell 10: Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=test_set.classes)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nConfusion Matrix:\n{cm}")

# %% — Cell 11: Score distribution histogram
fig, ax = plt.subplots(figsize=(8, 4))
fake_scores = all_probs[all_labels == 0]   # class 0 = fake
real_scores = all_probs[all_labels == 1]   # class 1 = real
ax.hist(real_scores, bins=30, alpha=0.6, label="Real faces", color="green")
ax.hist(fake_scores, bins=30, alpha=0.6, label="Fake faces", color="red")
ax.axvline(0.5, color="black", linestyle="--", label="Threshold (0.5)")
ax.set_xlabel("P(fake)")
ax.set_ylabel("Count")
ax.set_title("Score Distribution: Real vs Fake")
ax.legend()
plt.tight_layout()
plt.savefig("score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Cell 12: Save the trained model
SAVE_NAME = "video_clf_best.pth"
torch.save(best_model_state, SAVE_NAME)
file_size_mb = os.path.getsize(SAVE_NAME) / (1024 * 1024)
print(f"Model saved: {SAVE_NAME} ({file_size_mb:.1f} MB)")

# Copy to Google Drive
drive_path = "/content/drive/MyDrive/video_clf_best.pth"
import shutil
shutil.copy(SAVE_NAME, drive_path)
print(f"Copied to Google Drive: {drive_path}")

# Also save the plots
for fname in ["training_curves.png", "confusion_matrix.png",
              "score_distribution.png", "sample_batch.png"]:
    if os.path.isfile(fname):
        shutil.copy(fname, f"/content/drive/MyDrive/{fname}")

print("\n✅ All done! Download video_clf_best.pth and place it in:")
print("   backend/src/video/video_clf.pth")
