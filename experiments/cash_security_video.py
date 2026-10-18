"""Isolated real-video audit for R05. Never promotes diagnostic wrist hypotheses."""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experiments.counter_occupancy import ROOT, imwrite, read_json, sha, write_json

DATA = ROOT / "data/cash_security_v1"
META = ROOT / "datasets/cash_security_v1"


def infer():
    import torch

    from app.cv.inspect_cash import CashDetector
    from app.cv.train_cash import local_yolo

    assert torch.cuda.is_available()
    models = local_yolo()
    person = models(str(ROOT / "yolo11n.pt"))
    cash = CashDetector(ROOT / "data/cash/models/cash.pt", device=0)
    pose = models(str(ROOT / "data/counter_hand_v1/pretrained/yolo11s-pose.pt"))
    write_json(
        META / "model-audit.json",
        {
            "production_adapter": "app.cv.analyze_events: CashDetector",
            "cash_weights": "data/cash/models/cash.pt",
            "cash_sha256": cash.model_sha256,
            "cash_model_names": cash.model.names,
            "cash_selected_class_ids": cash.classes,
            "person_weights": "yolo11n.pt",
            "person_model_names": person.names,
            "id_card_supported_by_loaded_cash_model": "id_card" in cash.model.names.values(),
            "id_detector_validated": False,
            "production_replacement": False,
            "pose": "data/counter_hand_v1/pretrained/yolo11s-pose.pt",
            "device": torch.cuda.get_device_name(0),
        },
    )
    for s in read_json(META / "sources.json"):
        out = DATA / s["id"]
        out.mkdir(parents=True, exist_ok=True)
        if (out / "input-detections.json").exists() and (out / "poses.json").exists():
            continue
        assert sha(s["video"]) == s["video_sha256"]
        existing = ROOT / s["cache"] / "detections.json" if s.get("cache") else None
        old_poses = ROOT / "data/counter_hand_v1" / s["id"] / "poses.json"
        if existing and existing.exists() and old_poses.exists():
            write_json(out / "input-detections.json", read_json(existing))
            write_json(out / "poses.json", read_json(old_poses))
            print("Reused verified observations", s["id"], flush=True)
            continue
        cap = cv2.VideoCapture(s["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride, pose_stride = round(fps / 4), round(fps / 15)
        frames, poses, index = [], [], 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            h, w = image.shape[:2]
            if index % stride == 0:
                pred = person.predict(
                    image, classes=[0], conf=0.25, imgsz=640, device=0, verbose=False
                )[0]
                frames.append(
                    {
                        "frame_index": index,
                        "timestamp_sec": index / fps,
                        "width": w,
                        "height": h,
                        "people": [
                            {"box_xyxy": b, "confidence": c}
                            for b, c in zip(
                                pred.boxes.xyxy.cpu().tolist(),
                                pred.boxes.conf.cpu().tolist(),
                                strict=True,
                            )
                        ],
                        "objects": [
                            {
                                "label": o["label"],
                                "confidence": o["confidence"],
                                "bbox": [
                                    v / (w if n % 2 == 0 else h)
                                    for n, v in enumerate(o["box_xyxy"])
                                ],
                            }
                            for o in cash.detect(image)
                        ],
                    }
                )
            if index % pose_stride == 0:
                pred = pose.predict(image, conf=0.25, imgsz=1280, device=0, verbose=False)[0]
                poses.append(
                    {
                        "frame_index": index,
                        "timestamp_sec": index / fps,
                        "width": w,
                        "height": h,
                        "poses": [
                            {
                                "bbox_px": b,
                                "confidence": c,
                                "joints": {str(k): j[k] for k in range(5, 13)},
                            }
                            for b, c, j in zip(
                                pred.boxes.xyxy.cpu().tolist(),
                                pred.boxes.conf.cpu().tolist(),
                                pred.keypoints.data.cpu().tolist(),
                                strict=True,
                            )
                        ],
                    }
                )
                if len(poses) % 150 == 0:
                    print(s["id"], len(poses), "pose frames", flush=True)
            index += 1
        cap.release()
        write_json(
            out / "input-detections.json",
            {
                "video_sha256": s["video_sha256"],
                "fps": fps,
                "stride": stride,
                "person_weights_sha256": sha(ROOT / "yolo11n.pt"),
                "cash_weights_sha256": cash.model_sha256,
                "frames": frames,
            },
        )
        write_json(
            out / "poses.json",
            {
                "fps": fps,
                "stride": pose_stride,
                "duration_sec": index / fps,
                "source_video_sha256": s["video_sha256"],
                "frames": poses,
            },
        )
        print("Saved", s["id"], flush=True)


def replay():
    from app.cv.analyze_events import analyze_video

    for s in read_json(META / "sources.json"):
        out = DATA / s["id"]
        print(
            s["id"],
            analyze_video(
                video=__import__("pathlib").Path(s["video"]),
                output=out / "legacy",
                config_path=ROOT / "config/event_camera.json",
                replay=out / "input-detections.json",
            ),
            flush=True,
        )


def inspect():
    """Fixed interval contact sheets and zone motion, never used as labeled safe actions."""
    from experiments.counter_hand_evidence import annotate

    p = read_json(ROOT / "datasets/counter_hand_v1/protocol.json")
    security = read_json(ROOT / "config/cash_security_camera.json")
    for s in read_json(META / "sources.json"):
        out = DATA / s["id"]
        cap = cv2.VideoCapture(s["video"])
        fps, index, panels, page = cap.get(cv2.CAP_PROP_FPS), 0, [], 0
        step = round(fps)
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index % step == 0:
                imwrite(out / "review-frames" / f"{index:05d}.jpg", image)
                annotated = annotate(image, [], p)
                for name, color in [("safe", (255, 100, 0)), ("table", (255, 0, 255))]:
                    poly = np.rint(
                        np.array(security["zones"][name]) * [image.shape[1], image.shape[0]]
                    ).astype(np.int32)
                    cv2.polylines(annotated, [poly], True, color, 1)
                tile = Image.new("RGB", (232, 444), "white")
                tile.paste(
                    Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).resize((232, 424)),
                    (0, 20),
                )
                ImageDraw.Draw(tile).text((3, 3), f"{s['id']} {index / fps:.2f}s", fill="black")
                panels.append(tile)
                if len(panels) == 24:
                    sheet = Image.new("RGB", (232 * 6, 444 * 4), "white")
                    for n, tile in enumerate(panels):
                        sheet.paste(tile, ((n % 6) * 232, (n // 6) * 444))
                    sheet.save(out / f"review-{page}.jpg", quality=92)
                    panels, page = [], page + 1
            index += 1
        cap.release()
        if panels:
            sheet = Image.new("RGB", (232 * 6, 444 * math.ceil(len(panels) / 6)), "white")
            for n, tile in enumerate(panels):
                sheet.paste(tile, ((n % 6) * 232, (n // 6) * 444))
            sheet.save(out / f"review-{page}.jpg", quality=92)
        print("Review sheets", s["id"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["infer", "replay", "inspect"])
    args = parser.parse_args()
    {"infer": infer, "replay": replay, "inspect": inspect}[args.command]()
