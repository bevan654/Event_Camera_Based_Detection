# Draw YOLO crack boxes on a few event images to check the labels line up.
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def draw(img_path, label_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        _, cx, cy, bw, bh = map(float, line.split())
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(img, img_path.stem[-40:], (3, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="data/processed/preview.png")
    args = ap.parse_args()
    data = Path(args.data)
    images = sorted((data / "images").glob("*.png"))
    random.seed(0)
    picks = random.sample(images, min(args.n, len(images)))
    tiles = [draw(p, data / "labels" / f"{p.stem}.txt") for p in picks]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)  # pad so the grid is full
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(args.out, grid)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
