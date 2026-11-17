# Run a trained crack model over an event recording and save a video of the detections.
import argparse
import csv
from pathlib import Path

import cv2
import h5py
import numpy as np
from ultralytics import YOLO

from make_event_images import H, W, is_flooded, make_image, pick_window


def load_events(path):
    # supports v2e output (one "events" array of t,x,y,p) and ev-CIVIL files (separate arrays)
    with h5py.File(path, "r") as f:
        if "events" in f:
            e = f["events"][:]
            return e[:, 0].astype(np.int64), e[:, 1], e[:, 2], e[:, 3]
        return f["event_timestamp"][:].astype(np.int64), f["x"][:], f["y"][:], f["polarity"][:]


def fit_to_sensor(x, y, p, t):
    # centre-crop to 4:3 then scale down to the 346x260 sensor the model was trained on
    width, height = int(x.max()) + 1, int(y.max()) + 1
    crop_w = min(width, height * 4 // 3)
    x0 = (width - crop_w) // 2
    keep = (x >= x0) & (x < x0 + crop_w)
    scale = W / crop_w
    xs = ((x[keep] - x0) * scale).astype(np.int64)
    ys = (y[keep] * scale).astype(np.int64)
    ok = ys < H
    return t[keep][ok], xs[ok], ys[ok], p[keep][ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--weights", default="runs/field_only_n/weights/best.pt")
    ap.add_argument("--step-ms", type=float, default=100)  # one event image every 100 ms
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    t, x, y, p = load_events(args.events)
    if x.max() >= W or y.max() >= H:
        t, x, y, p = fit_to_sensor(x, y, p, t)
    out = Path("runs/predict") / (args.name or Path(args.events).parent.name)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    video = cv2.VideoWriter(str(out / "detections.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W * 2, H * 2))

    rows = []
    for anchor in np.arange(t[0], t[-1], args.step_ms * 1000):
        window = pick_window(t, x, y, anchor, 15_000)
        if window is None or is_flooded(x[window[0]:window[1]], y[window[0]:window[1]]):
            rows.append([round((anchor - t[0]) / 1e6, 2), "skipped", 0, 0])
            continue
        s, e = window
        img = make_image(x[s:e], y[s:e], p[s:e])[..., ::-1].copy()  # RGB -> BGR for OpenCV
        boxes = model.predict(img, conf=args.conf, iou=0.4, verbose=False)[0].boxes
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, f"{float(b.conf):.2f}", (x1 + 2, y1 + 10), 0, 0.35, (0, 255, 0), 1)
        sec = (anchor - t[0]) / 1e6
        cv2.putText(img, f"{sec:5.1f}s  boxes: {len(boxes)}", (3, 12), 0, 0.4, (0, 255, 255), 1)
        video.write(cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST))
        rows.append([round(sec, 2), e - s, len(boxes), round(float(boxes.conf.max()), 3) if len(boxes) else 0])
    video.release()

    with open(out / "detections.csv", "w", newline="") as f:
        csv.writer(f).writerows([["time_s", "num_events", "num_boxes", "max_conf"]] + rows)
    used = [r for r in rows if r[1] != "skipped"]
    hits = [r for r in used if r[2] > 0]
    print(f"{len(used)} event images ({len(rows) - len(used)} skipped), {len(hits)} with detections, {sum(r[2] for r in used)} boxes total")
    print(f"Saved {out / 'detections.mp4'} and {out / 'detections.csv'}")


if __name__ == "__main__":
    main()
