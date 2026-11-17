# ROS 2 node: listen to the simulator's event camera, run the crack model live, publish boxes and an annotated image.
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from lotusim_sensor_msgs.msg import EventArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

sys.path.insert(0, str(Path(__file__).parent))
from make_event_images import H, W, is_flooded, make_image, pick_window  # noqa: E402

BUFFER_US = 300_000  # keep the last 0.3 s of events


class CrackDetector(Node):
    def __init__(self, args):
        super().__init__("crack_detector")
        self.args = args
        self.model = YOLO(args.weights)
        self.chunks = []  # list of (t, x, y, p) arrays, newest last
        self.create_subscription(EventArray, args.topic, self.on_events, 50)
        self.image_pub = self.create_publisher(Image, "/crack_detector/image", 10)
        self.det_pub = self.create_publisher(Detection2DArray, "/crack_detector/detections", 10)
        self.create_timer(1 / args.rate, self.detect)
        self.get_logger().info(f"Listening on {args.topic}, model {args.weights}")

    def on_events(self, msg):
        # absolute event time in microseconds = message stamp + per-event offset
        stamp_us = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1000
        t = stamp_us + np.asarray(msg.t_offset_ns, np.int64) // 1000
        self.chunks.append((t, np.asarray(msg.x), np.asarray(msg.y), np.asarray(msg.polarity), msg.header))
        newest = t.max() if len(t) else stamp_us
        self.chunks = [c for c in self.chunks if len(c[0]) and c[0].max() > newest - BUFFER_US]

    def detect(self):
        if not self.chunks:
            return
        t, x, y, p = (np.concatenate([c[i] for c in self.chunks]) for i in range(4))
        order = np.argsort(t, kind="stable")
        t, x, y, p = t[order], x[order], y[order], p[order]
        window = pick_window(t, x, y, t[-1], self.args.min_ms * 1000)  # grows backwards from the newest event
        if window is None or is_flooded(x[window[0]:window[1]], y[window[0]:window[1]]):
            return
        s, e = window
        img = make_image(x[s:e], y[s:e], p[s:e])[..., ::-1].copy()  # RGB -> BGR for OpenCV
        boxes = self.model.predict(img, conf=self.args.conf, iou=0.4, verbose=False)[0].boxes

        if not rclpy.ok():
            return  # shutting down
        header = self.chunks[-1][4]
        dets = Detection2DArray(header=header)
        for b in boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            d = Detection2D(header=header)
            d.bbox.center.position.x, d.bbox.center.position.y = (x1 + x2) / 2, (y1 + y2) / 2
            d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = "crack", float(b.conf)
            d.results.append(hyp)
            dets.detections.append(d)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(img, f"{float(b.conf):.2f}", (int(x1) + 2, int(y1) + 10), 0, 0.35, (0, 255, 0), 1)
        cv2.putText(img, f"{(t[e - 1] - t[s]) / 1000:.0f} ms  {e - s} ev  cracks: {len(boxes)}", (3, 12), 0, 0.4, (0, 255, 255), 1)
        self.det_pub.publish(dets)
        self.image_pub.publish(Image(header=header, height=H, width=W, encoding="bgr8", step=W * 3, data=img.tobytes()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/energy/bluerov_heavy_eventclone/event_camera/events")
    ap.add_argument("--weights", default="runs/field_lab_n/weights/best.pt")
    ap.add_argument("--rate", type=float, default=10)  # detections per second
    ap.add_argument("--conf", type=float, default=0.4)  # stricter than the paper's 0.2 to cut false alarms on water
    ap.add_argument("--min-ms", type=float, default=15)  # Algorithm 1 minimum window (use 30 for dark scenes)
    args = ap.parse_args()
    rclpy.init()
    node = CrackDetector(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
