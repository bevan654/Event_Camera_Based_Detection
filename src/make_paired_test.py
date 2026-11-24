# Build a paired test set: for each test event image, the real camera (APS) frame from the same moment, same labels.
import argparse
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

from make_event_images import yolo_lines

SPLITS = ["test_day", "test_lowlight"]


def nearest_frame(seq_dir, t_us, cache):
    # load a recording's frames once, return (frame, time gap in ms) for the frame closest to t_us
    if seq_dir not in cache:
        with h5py.File(seq_dir / f"{seq_dir.name}_frames.h5") as h:
            cache.clear()  # keep only one recording in memory
            cache[seq_dir] = (h["frames"][:], h["frame_timestamp"][:].astype(np.int64))
    frames, ts = cache[seq_dir]
    i = int(np.argmin(np.abs(ts - t_us)))
    return frames[i], abs(ts[i] - t_us) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="data/yolo_40")
    ap.add_argument("--samples", default="data/processed_40/samples.csv")
    ap.add_argument("--raw", default="data/evCIVIL_Field_dataset")
    ap.add_argument("--out", default="data/paired")
    ap.add_argument("--max-gap-ms", type=float, default=5.0)  # skip if no frame this close to the event image
    args = ap.parse_args()
    yolo, out = Path(args.yolo), Path(args.out)
    samples = pd.read_csv(args.samples).set_index("image")
    cache = {}

    for split in SPLITS:
        kept, own, far, no_label = 0, 0, 0, 0
        for kind in ["events", "frames"]:
            for sub in ["images", "labels"]:
                (out / kind / sub / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((yolo / "images" / split).glob("*.png")):
            seq_dir = Path(args.raw) / samples.loc[img.stem, "sequence"]
            t_us = samples.loc[img.stem, "timestamp_us"]
            frame, gap = nearest_frame(seq_dir, t_us, cache)
            if gap > args.max_gap_ms:
                far += 1
                continue
            label = yolo / "labels" / split / f"{img.stem}.txt"
            frame_label = label.read_text()
            fl_file = seq_dir / f"{seq_dir.name}_frame_label.npy"
            if fl_file.exists():  # authors labelled the frames separately here, use their frame boxes
                fl = np.load(fl_file)
                rows = fl[(np.abs(fl[:, 0] - t_us) < 1000) & (fl[:, 1] == 0)]
                if len(rows) == 0:
                    no_label += 1
                    continue
                frame_label = "".join(l + "\n" for l in yolo_lines(rows[:, 2:]))
                own += 1
            cv2.imwrite(str(out / "frames" / "images" / split / img.name), frame)
            shutil.copy(img, out / "events" / "images" / split / img.name)
            shutil.copy(label, out / "events" / "labels" / split / label.name)
            (out / "frames" / "labels" / split / label.name).write_text(frame_label)
            kept += 1
        print(f"{split}: kept {kept} ({own} with separate frame labels), skipped {far} (no frame within {args.max_gap_ms} ms), {no_label} (no frame label)")

    for kind in ["events", "frames"]:
        for split in SPLITS:
            p = (out / kind).resolve()
            (out / kind / f"data_{split}.yaml").write_text(f"path: {p}\ntrain: images/{split}\nval: images/{split}\ntest: images/{split}\nnames:\n  0: crack\n")


if __name__ == "__main__":
    main()
