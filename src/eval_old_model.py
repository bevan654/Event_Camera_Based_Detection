# Score the old grayscale+CLAHE crack model (trained on normal photos) on our event-image test sets.
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

OLD_MODEL = "/home/shaj0038/Desktop/crack_detection_model/crack.v1i.yolov11/runs/segment/v1_img832/weights/best.pt"
TESTS = {"well-lit": "test_day", "low-light": "test_lowlight"}


def clahe(gray):
    # same steps as the old project's apply_clahe.py: CLAHE -> sharpen
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.addWeighted(c, 2, cv2.GaussianBlur(c, (13, 13), 0), -1, 0)


def gray_b(img):
    # grayscale B: mid-grey background, ON events lighter, OFF events darker (saved images are BGR = [ON+OFF, OFF, ON])
    on, off = img[..., 2].astype(np.float32) / 255, img[..., 1].astype(np.float32) / 255
    return np.clip(128 + 127 * (on - off), 0, 255).astype(np.uint8)


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


MODES = {
    "rgb": lambda img: img,
    "gray_clahe": lambda img: to_bgr(clahe(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
    "gray_b": lambda img: to_bgr(gray_b(img)),
    "gray_b_clahe": lambda img: to_bgr(clahe(gray_b(img))),
}


def box_to_polygon(line):
    # the old model is a segmentation model, so write each box as a 4-corner outline (box scores are unchanged)
    c, cx, cy, w, h = line.split()
    x1, y1, x2, y2 = float(cx) - float(w) / 2, float(cy) - float(h) / 2, float(cx) + float(w) / 2, float(cy) + float(h) / 2
    return f"{c} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"


def build(src, out, mode):
    # write preprocessed copies of both test sets plus labels, return the yaml path per test set
    yamls = {}
    for name, split in TESTS.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted((src / "labels" / split).glob("*.txt")):
            (lbl_dir / f.name).write_text("".join(box_to_polygon(l) + "\n" for l in f.read_text().splitlines() if l.strip()))
        for f in sorted((src / "images" / split).glob("*.png")):
            cv2.imwrite(str(img_dir / f.name), MODES[mode](cv2.imread(str(f))))
        yamls[name] = out / f"data_{split}.yaml"
        yamls[name].write_text(f"path: {out.resolve()}\ntrain: images/{split}\nval: images/{split}\ntest: images/{split}\nnames:\n  0: crack\n")
    return yamls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=OLD_MODEL)
    ap.add_argument("--data", default="data/yolo")
    ap.add_argument("--out", default="data/old_model_eval")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    args = ap.parse_args()
    model = YOLO(args.weights)
    rows = []
    for mode in args.modes:
        for name, yaml in build(Path(args.data), Path(args.out) / mode, mode).items():
            b = model.val(data=str(yaml), split="test", imgsz=640, conf=args.conf, iou=args.iou, plots=False, verbose=False).box
            f1 = 2 * b.mp * b.mr / max(b.mp + b.mr, 1e-9)
            rows.append(f"{mode:11s} {name:10s} {b.map50:6.3f} {b.map:9.3f} {b.mp:6.3f} {b.mr:6.3f} {f1:6.3f}")
    print(f"\n{args.weights}  (conf {args.conf}, iou {args.iou})")
    print("input       test set    mAP50  mAP50-95   prec recall     F1")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
