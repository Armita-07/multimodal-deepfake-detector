import os
from collections import defaultdict
from src.image.predict import predict_image

ROOT = "test_images"
EXT = (".jpg", ".jpeg", ".png", ".webp")
rows = []

for folder, _, files in os.walk(ROOT):
    for name in sorted(files):
        if not name.lower().endswith(EXT):
            continue
        path = os.path.join(folder, name)
        low_name = name.lower()
        sub = os.path.relpath(folder, ROOT)          # "." if directly in test_images
        low_sub = sub.lower()

        # 1) work out the true answer: file name first, then folder name
        if low_name.startswith("real_"):
            truth = "real"
        elif low_name.startswith("fake_"):
            truth = "fake"
        elif "fake" in low_sub:
            truth = "fake"
        elif "real" in low_sub:
            truth = "real"
        else:
            truth = None                              # e.g. a "Non-faces" folder

        # 2) work out the source (group label)
        parts = os.path.splitext(name)[0].split("_")
        if truth and low_name.startswith(("real_", "fake_")) and len(parts) > 1:
            source = parts[1]
        else:
            source = sub if sub != "." else "root"

        try:
            p = predict_image(path)
        except Exception as e:
            print(f"SKIPPED (could not read): {os.path.join(sub, name)}  [{e}]")
            continue

        guess = "fake" if p >= 0.5 else "real"
        rows.append((os.path.join(sub, name), truth, source, p, guess))
        if truth is None:
            status = "not scored"
        else:
            status = "OK" if guess == truth else "WRONG"
        print(f"{os.path.join(sub, name):55s} truth={str(truth):5s} fake_prob={p:.3f} guess={guess:5s} {status}")

print("\nSummary by group:")
groups = defaultdict(list)
for rel, truth, source, p, guess in rows:
    if truth:
        groups[(truth, source)].append(guess == truth)
for (truth, source), res in sorted(groups.items()):
    print(f"  {truth:5s} {source:30s} correct {sum(res)}/{len(res)}")

scored = [(t, g) for _, t, _, _, g in rows if t]
print(f"\nOverall: {sum(t == g for t, g in scored)}/{len(scored)} correct")