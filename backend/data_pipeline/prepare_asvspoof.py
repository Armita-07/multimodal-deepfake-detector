import argparse
import os
import shutil

PROTOCOL_MAP = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

FLAC_DIR_MAP = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev": "ASVspoof2019_LA_dev/flac",
    "eval": "ASVspoof2019_LA_eval/flac",
}


def parse_protocol(proto_path):
    entries = []
    with open(proto_path) as f:
        for line in f:
            parts = line.strip().split()
            speaker_id = parts[0]
            utt_id = parts[1]
            label = parts[-1]
            entries.append((speaker_id, utt_id, label))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to LA directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    for split in ("train", "dev", "eval"):
        proto_dir = os.path.join(args.root, "ASVspoof2019_LA_cm_protocols")
        if not os.path.isdir(proto_dir):
            proto_dir = os.path.join(args.root, "LA_cm_protocols")
        proto_path = os.path.join(proto_dir, PROTOCOL_MAP[split])

        flac_dir = os.path.join(args.root, FLAC_DIR_MAP[split])
        if not os.path.isdir(flac_dir):
            alt_flac_dir = os.path.join(args.root, f"ASVspoof2019_LA_{split}")
            if os.path.isdir(alt_flac_dir):
                flac_dir = alt_flac_dir

        entries = parse_protocol(proto_path)
        counts = {"real": 0, "fake": 0}
        for _, utt_id, label in entries:
            cls = "real" if label == "bonafide" else "fake"

            if args.max_per_class and counts[cls] >= args.max_per_class:
                continue

            src = os.path.join(flac_dir, f"{utt_id}.flac")
            if not os.path.isfile(src):
                continue

            dst_dir = os.path.join(args.out, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, f"{utt_id}.flac"))
            counts[cls] += 1

        print(f"{split}: real={counts['real']}, fake={counts['fake']}")

    print("Done.")


if __name__ == "__main__":
    main()
