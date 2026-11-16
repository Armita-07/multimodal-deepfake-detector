import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=False, post_process=False, select_largest=True, device=device)


def extract_faces_from_videos(input_dir, output_dir, video_list, num_frames=24):
    os.makedirs(output_dir, exist_ok=True)
    for vid_name in tqdm(video_list):
        cap = cv2.VideoCapture(os.path.join(input_dir, vid_name))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue

        indices = np.linspace(0, total - 1, num_frames).astype(int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is None or len(boxes) == 0:
                continue

            x1, y1, x2, y2 = boxes[0]
            margin = 0.3 * max(x2 - x1, y2 - y1)
            x1 = max(0, int(x1 - margin))
            y1 = max(0, int(y1 - margin))
            x2 = min(rgb.shape[1], int(x2 + margin))
            y2 = min(rgb.shape[0], int(y2 + margin))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = pil_img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)
            crop.save(os.path.join(output_dir, f"{os.path.splitext(vid_name)[0]}_f{idx:04d}.png"))
        cap.release()


def main():
    splits_config = [
        ("real", "original_sequences/youtube/c23/videos"),
        ("fake", "manipulated_sequences/Deepfakes/c23/videos"),
    ]

    base_dir = "ff_data"
    for label, rel_path in splits_config:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.isdir(full_path):
            continue

        videos = sorted([f for f in os.listdir(full_path) if f.endswith(".mp4")])
        splits = {
            "train": videos[:70],
            "val": videos[70:85],
            "test": videos[85:100],
        }

        for split_name, split_vids in splits.items():
            out_dir = os.path.join("dataset", split_name, label)
            print(f"Extracting {split_name}/{label}...")
            extract_faces_from_videos(full_path, out_dir, split_vids)


if __name__ == "__main__":
    main()
