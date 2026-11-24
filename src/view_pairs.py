# Slide through the paired test set: real camera frame on the left, event image on the right, crack boxes in yellow.
import argparse
from pathlib import Path

import cv2
import numpy as np

from eval_old_model import clahe

KEYS_NEXT, KEYS_PREV = {ord("d"), 65363, 16777236}, {ord("a"), 65361, 16777234}  # d / right arrow, a / left arrow (GTK and Qt codes)


def draw(img, label_file, scale):
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    for line in label_file.read_text().splitlines():
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_lowlight", choices=["test_day", "test_lowlight"])
    ap.add_argument("--data", default="data/paired")
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()
    root = Path(args.data)
    names = sorted(f.name for f in (root / "frames" / "images" / args.split).glob("*.png"))
    i, use_clahe = 0, False
    print("d / right arrow: next   a / left arrow: back   c: CLAHE on the frame on/off   q / Esc: quit")
    while True:
        n, stem = names[i], names[i][:-4]
        frame = cv2.imread(str(root / "frames" / "images" / args.split / n), cv2.IMREAD_GRAYSCALE)
        frame = cv2.cvtColor(clahe(frame) if use_clahe else frame, cv2.COLOR_GRAY2BGR)
        events = cv2.imread(str(root / "events" / "images" / args.split / n))
        left = draw(frame, root / "frames" / "labels" / args.split / f"{stem}.txt", args.scale)
        right = draw(events, root / "events" / "labels" / args.split / f"{stem}.txt", args.scale)
        view = np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right])
        cv2.putText(view, f"{i + 1}/{len(names)}  frame{' + CLAHE' if use_clahe else ''}", (10, 30), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(view, "events", (left.shape[1] + 16, 30), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(view, stem.split("__", 1)[1], (10, view.shape[0] - 12), 0, 0.5, (255, 255, 255), 1)
        cv2.imshow("pairs", view)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27) or cv2.getWindowProperty("pairs", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in KEYS_NEXT:
            i = (i + 1) % len(names)
        elif key in KEYS_PREV:
            i = (i - 1) % len(names)
        elif key == ord("c"):
            use_clahe = not use_clahe
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
