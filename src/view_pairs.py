# Slide through the paired test set: real camera frame on the left, events on the right, crack boxes in yellow.
# Space plays the recording live around the current moment: frames on the left, a sliding event window on the right.
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

from eval_old_model import clahe
from make_event_images import load_events, make_image

KEYS_NEXT, KEYS_PREV = {ord("d"), 65363, 16777236}, {ord("a"), 65361, 16777234}  # d / right arrow, a / left arrow (GTK and Qt codes)
STEP_US = 33_000  # playback advances ~30 steps per second of recording


def boxes_from_lines(lines):
    # YOLO "0 cx cy w h" lines -> list of (cx, cy, w, h) scaled 0-1
    return [tuple(map(float, l.split()[1:])) for l in lines if l.strip()]


def draw(img, boxes, scale):
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    for cx, cy, bw, bh in boxes:
        cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 255), 2)
    return img


def show(left, right, top_left, bottom):
    view = np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right])
    cv2.putText(view, top_left, (10, 30), 0, 0.8, (255, 255, 255), 2)
    cv2.putText(view, "events", (left.shape[1] + 16, 30), 0, 0.8, (255, 255, 255), 2)
    cv2.putText(view, bottom, (10, view.shape[0] - 12), 0, 0.5, (255, 255, 255), 1)
    cv2.imshow("pairs", view)


def load_recording(seq_dir):
    # everything needed to play one recording: events, frames and crack labels (COCO px -> YOLO 0-1)
    t, x, y, p = load_events(seq_dir / f"{seq_dir.name}_events.h5")
    with h5py.File(seq_dir / f"{seq_dir.name}_frames.h5") as h:
        frames, ft = h["frames"][:], h["frame_timestamp"][:].astype(np.int64)
    lab = np.load(seq_dir / f"{seq_dir.name}_label.npy")
    lab = lab[lab[:, 1] == 0]
    return t, x, y, p, frames, ft, lab


def label_boxes(lab, t_now, max_gap_us=50_000):
    # crack boxes from the label time closest to t_now (none if no label nearby)
    if len(lab) == 0 or np.abs(lab[:, 0] - t_now).min() > max_gap_us:
        return []
    rows = lab[lab[:, 0] == lab[np.argmin(np.abs(lab[:, 0] - t_now)), 0]]
    return [((x + w / 2) / 346, (y + h / 2) / 260, w / 346, h / 260) for x, y, w, h in rows[:, 2:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_lowlight", choices=["test_day", "test_lowlight"])
    ap.add_argument("--data", default="data/paired")
    ap.add_argument("--samples", default="data/processed_40/samples.csv")
    ap.add_argument("--raw", default="data/evCIVIL_Field_dataset")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=0.5)  # playback speed (1 = real time)
    args = ap.parse_args()
    root = Path(args.data)
    samples = pd.read_csv(args.samples).set_index("image")
    names = sorted(f.name for f in (root / "frames" / "images" / args.split).glob("*.png"))
    i, use_clahe, playing, rec, rec_dir = 0, False, False, None, None
    print("d / right: next   a / left: back   space: play/pause live events   c: CLAHE on/off   q / Esc: quit")
    while True:
        n, stem = names[i], names[i][:-4]
        prep = lambda g: cv2.cvtColor(clahe(g) if use_clahe else g, cv2.COLOR_GRAY2BGR)
        tag = f"{i + 1}/{len(names)}  frame{' + CLAHE' if use_clahe else ''}"
        if not playing:
            frame = cv2.imread(str(root / "frames" / "images" / args.split / n), cv2.IMREAD_GRAYSCALE)
            events = cv2.imread(str(root / "events" / "images" / args.split / n))
            fb = boxes_from_lines((root / "frames" / "labels" / args.split / f"{stem}.txt").read_text().splitlines())
            eb = boxes_from_lines((root / "events" / "labels" / args.split / f"{stem}.txt").read_text().splitlines())
            show(draw(prep(frame), fb, args.scale), draw(events, eb, args.scale), tag, stem.split("__", 1)[1] + "   (still)")
            key = cv2.waitKeyEx(0)
        else:
            t, x, y, p, frames, ft, lab = rec
            win_us = 30_000 if samples.loc[stem, "light"] == "night" else 15_000
            s, e = np.searchsorted(t, [t_now - win_us, t_now])
            events = make_image(x[s:e], y[s:e], p[s:e])[..., ::-1] if e > s else np.zeros((260, 346, 3), np.uint8)
            frame = frames[np.searchsorted(ft, t_now, side="right") - 1] if t_now >= ft[0] else frames[0]  # latest frame so far
            boxes = label_boxes(lab, t_now)
            show(draw(prep(frame), boxes, args.scale), draw(events, boxes, args.scale), tag,
                 f"{stem.split('__', 1)[1]}   LIVE t={t_now / 1e6:.3f}s  {win_us // 1000} ms window  x{args.speed}")
            key = cv2.waitKeyEx(max(1, int(STEP_US / 1000 / args.speed)))
            t_now += STEP_US
            if t_now > t[-1]:
                t_now = t[0] + win_us  # loop the recording
        if key in (ord("q"), 27) or cv2.getWindowProperty("pairs", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in KEYS_NEXT or key in KEYS_PREV:
            i = (i + (1 if key in KEYS_NEXT else -1)) % len(names)
            playing = False
        elif key == ord("c"):
            use_clahe = not use_clahe
        elif key == ord(" "):
            playing = not playing
            if playing:
                seq_dir = Path(args.raw) / samples.loc[stem, "sequence"]
                if seq_dir != rec_dir:
                    rec, rec_dir = load_recording(seq_dir), seq_dir
                t_now = int(samples.loc[stem, "timestamp_us"]) - 1_000_000  # start 1 s before the labelled moment
                t_now = max(t_now, int(rec[0][0]) + 30_000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
