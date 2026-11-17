# Train a YOLO11 crack detector on the event images.
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")  # COCO-pretrained starting point
    ap.add_argument("--data", default="data/yolo/data_test_day.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="smoke_test")
    args = ap.parse_args()

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        project=str(Path("runs").resolve()),  # absolute, otherwise Ultralytics nests it under runs/detect/
        name=args.name,
        exist_ok=True,
        hsv_h=0, hsv_s=0, hsv_v=0,  # colour jitter would scramble the ON/OFF channels
        flipud=0.5,  # vertical flips, as in the paper
        seed=0,
    )


if __name__ == "__main__":
    main()
