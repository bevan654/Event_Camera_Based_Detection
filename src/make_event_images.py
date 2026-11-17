# Turn ev-CIVIL event recordings into 3-channel event images + YOLO crack labels.
import argparse
import csv
from pathlib import Path

import cv2
import h5py
import numpy as np

W, H = 346, 260
NIGHT_WORDS = ("night", "tunnel")  # folders recorded in low light use the longer window


def load_events(path):
    with h5py.File(path, "r") as f:
        return f["event_timestamp"][:].astype(np.int64), f["x"][:], f["y"][:], f["polarity"][:]


def pick_window(t, x, y, anchor, min_us, max_us=100_000, q=100, extra=175):
    # Algorithm 1: grow around the anchor until long enough AND one 4x4 grid cell stands out
    i = np.searchsorted(t, anchor)
    step = 1
    while True:
        s, e = max(i - q * step, 0), min(i + q * step, len(t))
        span = t[e - 1] - t[s]
        cells = np.bincount((y[s:e] * 4 // H) * 4 + x[s:e] * 4 // W, minlength=16)
        if span > min_us and (cells - cells.mean()).max() > extra:
            return s, e
        if span > max_us or (s == 0 and e == len(t)):
            return None  # no clear structure found: crack not visible
        step += 1


def is_flooded(x, y, limit=0.9):
    # more than 90% of pixels firing means a sensor glitch, not a real scene
    return len(np.unique(y.astype(np.int64) * W + x)) > limit * W * H


def make_image(x, y, p):
    # count ON and OFF events per pixel, clip outliers at 3 std, scale to 0-255
    hist = np.zeros((2, H, W), np.float32)
    np.add.at(hist, (1 - p.astype(np.int64), y, x), 1)
    hist = np.clip(hist, 0, 3 * hist.std())
    hist /= max(hist.max(), 1e-9)
    on, off = hist
    rgb = np.dstack([on, off, np.clip(on + off, 0, 1)])
    return (rgb * 255).astype(np.uint8)


def yolo_lines(boxes):
    # COCO [x, y, w, h] in pixels -> YOLO "0 cx cy w h" scaled to 0-1
    lines = []
    for x, y, w, h in boxes:
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def process_sequence(seq_dir, root, out, per_seq):
    name = seq_dir.name
    labels = np.load(seq_dir / f"{name}_label.npy")
    cracks = labels[labels[:, 1] == 0]
    if len(cracks) == 0:
        return []
    times = np.unique(cracks[:, 0])
    anchors = times[np.linspace(0, len(times) - 1, min(per_seq, len(times))).astype(int)]  # evenly spread moments
    rel = seq_dir.relative_to(root)
    night = any(w in str(rel).lower() for w in NIGHT_WORDS)
    t, x, y, p = load_events(seq_dir / f"{name}_events.h5")
    rows = []
    for anchor in anchors:
        window = pick_window(t, x, y, anchor, 30_000 if night else 15_000)
        if window is None or is_flooded(x[window[0]:window[1]], y[window[0]:window[1]]):
            print(f"  skipped {rel} @ {int(anchor)}")
            continue
        s, e = window
        stem = f"{'__'.join(rel.parts)}__{int(anchor)}"
        cv2.imwrite(str(out / "images" / f"{stem}.png"), make_image(x[s:e], y[s:e], p[s:e])[..., ::-1])  # RGB -> BGR for OpenCV
        lines = yolo_lines(cracks[cracks[:, 0] == anchor][:, 2:])
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
        rows.append([stem, str(rel), int(anchor), round((t[e - 1] - t[s]) / 1000, 1), e - s, "night" if night else "day", len(lines)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/evCIVIL_Field_dataset/Field_dataset")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--per-seq", type=int, default=12)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for ev in sorted(root.rglob("*_events.h5")):
        rows += process_sequence(ev.parent, root, out, args.per_seq)
        print(f"{ev.parent.relative_to(root)}: total images {len(rows)}")
    with open(out / "samples.csv", "w", newline="") as f:
        csv.writer(f).writerows([["image", "sequence", "timestamp_us", "window_ms", "num_events", "light", "num_cracks"]] + rows)
    print(f"Done: {len(rows)} images in {out}")


if __name__ == "__main__":
    main()
