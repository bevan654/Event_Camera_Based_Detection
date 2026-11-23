# Score the old grayscale+CLAHE crack model (trained on normal photos) on our event-image test sets.
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

OLD_MODEL = "/home/shaj0038/Desktop/crack_detection_model/crack.v1i.yolov11/runs/segment/v1_img832/weights/best.pt"
TESTS = {"well-lit": "test_day", "low-light": "test_lowlight"}


def clahe_pipeline(img):
    # same steps as the old project's apply_clahe.py: grayscale -> CLAHE -> sharpen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp = cv2.addWeighted(c, 2, cv2.GaussianBlur(c, (13, 13), 0), -1, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


MODES = {"rgb": lambda img: img, "gray_clahe": clahe_pipeline}


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
    args = ap.parse_args()
    model = YOLO(args.weights)
    rows = []
    for mode in MODES:
        for name, yaml in build(Path(args.data), Path(args.out) / mode, mode).items():
            b = model.val(data=str(yaml), split="test", imgsz=640, conf=args.conf, iou=args.iou, plots=False, verbose=False).box
            f1 = 2 * b.mp * b.mr / max(b.mp + b.mr, 1e-9)
            rows.append(f"{mode:11s} {name:10s} {b.map50:6.3f} {b.map:9.3f} {b.mp:6.3f} {b.mr:6.3f} {f1:6.3f}")
    print(f"\n{args.weights}  (conf {args.conf}, iou {args.iou})")
    print("input       test set    mAP50  mAP50-95   prec recall     F1")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
