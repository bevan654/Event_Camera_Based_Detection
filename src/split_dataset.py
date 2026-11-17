# Split event images into train/val/test folders using the paper's official recording lists.
import argparse
import csv
import random
import shutil
from pathlib import Path

SPLIT_FILES = {"test_day": "splits/test_day.txt", "test_lowlight": "splits/test_lowlight.txt"}


def key(seq):
    # official lists skip some folder levels, so match on dataset + last two parts, e.g. "Field_dataset/dset_1/crack_41"
    parts = seq.strip().split("/")
    return "/".join([parts[0]] + parts[-2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--field-only", action="store_true")  # leave the lab recordings out of training
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    test_of = {key(line): name for name, f in SPLIT_FILES.items() for line in open(f) if line.strip()}
    rows = list(csv.DictReader(open(data / "samples.csv")))
    if args.field_only:
        rows = [r for r in rows if r["sequence"].startswith("Field_dataset/")]
    train_seqs = sorted({r["sequence"] for r in rows if key(r["sequence"]) not in test_of})
    random.seed(0)
    val_seqs = set(random.sample(train_seqs, max(1, round(len(train_seqs) * args.val_frac))))  # whole recordings go to val

    shutil.rmtree(out, ignore_errors=True)
    counts = {}
    for r in rows:
        split = test_of.get(key(r["sequence"])) or ("val" if r["sequence"] in val_seqs else "train")
        for kind, ext in (("images", "png"), ("labels", "txt")):
            dst = out / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(data / kind / f"{r['image']}.{ext}", dst)
        counts.setdefault(split, [set(), 0])
        counts[split][0].add(r["sequence"])
        counts[split][1] += 1

    for test in SPLIT_FILES:
        yaml = f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/{test}\nnames:\n  0: crack\n"
        (out / f"data_{test}.yaml").write_text(yaml)
    for split, (seqs, n) in sorted(counts.items()):
        print(f"{split:14s} {len(seqs):3d} recordings  {n:4d} images")


if __name__ == "__main__":
    main()
