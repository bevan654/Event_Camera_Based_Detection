# Score a trained model on both test sets and compare with the paper's crack results.
import argparse

from ultralytics import YOLO

TEST_SETS = {"well-lit": "data_test_day.yaml", "low-light": "data_test_lowlight.yaml"}
PAPER = {"well-lit": (0.48, 0.28, 0.57), "low-light": (0.46, 0.24, 0.60)}  # YOLOv6m events, crack: mAP50, mAP50-95, F1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/smoke_test/weights/best.pt")
    ap.add_argument("--data", default="data/yolo")  # folder holding the two test yaml files
    ap.add_argument("--conf", type=float, default=0.2)  # paper's test confidence threshold
    ap.add_argument("--iou", type=float, default=0.4)  # paper's NMS overlap threshold
    args = ap.parse_args()

    model = YOLO(args.weights)
    rows = []
    for name, yaml in TEST_SETS.items():
        m = model.val(data=f"{args.data}/{yaml}", split="test", conf=args.conf, iou=args.iou, plots=False, verbose=False).box
        f1 = 2 * m.mp * m.mr / max(m.mp + m.mr, 1e-9)
        rows.append(f"{name:10s} {m.map50:6.3f} {m.map:9.3f} {m.mp:6.3f} {m.mr:6.3f} {f1:6.3f}   {PAPER[name][0]:.2f} / {PAPER[name][1]:.2f} / {PAPER[name][2]:.2f}")
    print(f"\n{args.weights}  (conf {args.conf}, iou {args.iou})")
    print(f"{'test set':10s} {'mAP50':>6s} {'mAP50-95':>9s} {'prec':>6s} {'recall':>6s} {'F1':>6s}   paper (mAP50 / mAP50-95 / F1)")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
