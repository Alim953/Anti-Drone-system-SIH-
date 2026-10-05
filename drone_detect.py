#!/usr/bin/env python3
"""
Drone detection + tracking with YOLO (Ultralytics).

Train:   python drone_detect.py train --data drone.yaml --model yolov8s.pt --epochs 100
Detect:  python drone_detect.py detect --weights runs/detect/drone/weights/best.pt --source 0
         python drone_detect.py detect --weights best.pt --source video.mp4 --save
         python drone_detect.py detect --weights best.pt --source rtsp://ip:554/stream --no-show

Install: pip install ultralytics opencv-python

drone.yaml example:
    path: datasets/drone
    train: images/train
    val: images/val
    names:
      0: drone
"""
import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


def train(args):
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,   # drones are small objects -> use 960 or 1280
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name="drone",
        patience=30,
        mosaic=1.0,
        scale=0.5,
        degrees=5.0,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
    )
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")


def pixel_to_angle(cx, cy, w, h, hfov_deg):
    """Pixel offset from image centre -> azimuth/elevation (deg).
    Positive az = right of centre, positive el = above centre.
    Use these as error inputs for a gimbal/PTZ controller."""
    vfov_deg = hfov_deg * h / w
    az = (cx - w / 2) / w * hfov_deg
    el = -(cy - h / 2) / h * vfov_deg
    return az, el


def detect(args):
    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source

    log_file = open(args.log, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["time", "track_id", "conf", "x1", "y1", "x2", "y2",
                  "az_deg", "el_deg", "speed_px_s", "alert"])

    trails = defaultdict(lambda: deque(maxlen=40))   # track_id -> [(t, cx, cy)]
    seen = defaultdict(int)                          # track_id -> frames seen
    writer = None
    prev_t, fps = time.time(), 0.0

    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    try:
        for r in results:
            frame = r.orig_img.copy()
            h, w = frame.shape[:2]
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                ids = (r.boxes.id.int().cpu().tolist()
                       if r.boxes.id is not None else [-1] * len(boxes))

                for (x1, y1, x2, y2), conf, tid in zip(boxes, confs, ids):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    az, el = pixel_to_angle(cx, cy, w, h, args.hfov)

                    # speed from track history (pixels/second)
                    trails[tid].append((now, cx, cy))
                    speed = 0.0
                    if len(trails[tid]) >= 5:
                        t0, x0, y0 = trails[tid][0]
                        dt = now - t0
                        if dt > 0:
                            speed = ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5 / dt

                    # alert only after N consecutive tracked frames -> fewer false alarms
                    seen[tid] += 1
                    alert = seen[tid] >= args.confirm
                    colour = (0, 0, 255) if alert else (0, 200, 255)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
                    label = f"ID{tid} drone {conf:.2f} az{az:+.1f} el{el:+.1f}"
                    cv2.putText(frame, label, (int(x1), max(15, int(y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                    pts = [(int(x), int(y)) for _, x, y in trails[tid]]
                    for a, b in zip(pts, pts[1:]):
                        cv2.line(frame, a, b, colour, 1)

                    log.writerow([f"{now:.3f}", tid, f"{conf:.3f}",
                                  int(x1), int(y1), int(x2), int(y2),
                                  f"{az:.2f}", f"{el:.2f}", f"{speed:.1f}", int(alert)])
                    if alert and seen[tid] == args.confirm:
                        print(f"[ALERT] drone track {tid} confirmed  az={az:+.1f}  el={el:+.1f}")

            cv2.putText(frame, f"FPS {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if args.save:
                if writer is None:
                    Path("output").mkdir(exist_ok=True)
                    writer = cv2.VideoWriter("output/result.mp4",
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             25, (w, h))
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("Drone Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        log_file.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="YOLO drone detection system")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", default="yolov8s.pt")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=960)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default="0")   # "cpu" if no GPU

    d = sub.add_parser("detect")
    d.add_argument("--weights", required=True)
    d.add_argument("--source", default="0")   # webcam id, video file or RTSP url
    d.add_argument("--conf", type=float, default=0.35)
    d.add_argument("--imgsz", type=int, default=960)
    d.add_argument("--device", default="0")
    d.add_argument("--hfov", type=float, default=60.0, help="camera horizontal FOV (deg)")
    d.add_argument("--confirm", type=int, default=5, help="frames before alert")
    d.add_argument("--log", default="detections.csv")
    d.add_argument("--save", action="store_true")
    d.add_argument("--no-show", action="store_true")

    args = p.parse_args()
    train(args) if args.cmd == "train" else detect(args)


if __name__ == "__main__":
    main()
