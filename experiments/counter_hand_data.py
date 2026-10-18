"""Read-only pretrained wrist inference for the isolated counter diagnostic."""

from __future__ import annotations

import argparse
import time

import cv2

from experiments.counter_occupancy import ROOT, read_json, sha, write_json

DATA = ROOT / "data/counter_hand_v1"
META = ROOT / "datasets/counter_hand_v1"
WEIGHTS = DATA / "pretrained/yolo11s-pose.pt"


def prepare():
    if (META / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen")
    sources = read_json(ROOT / "datasets/counter_sequence_v1/sources.json")
    camera = read_json(ROOT / "config/event_camera.json")
    protected = set(ROOT.glob("app/**/*.py")) | set(ROOT.glob("config/**/*"))
    protected |= {p for p in ROOT.rglob("*.pt") if ".venv" not in p.parts and p != WEIGHTS}
    protected |= set(ROOT.glob("data/events/**/*"))
    protected |= set(ROOT.glob("experiments/*.py"))
    protected |= set(ROOT.glob("datasets/counter_sequence_v1/*"))
    protected.discard(ROOT / "experiments/counter_hand_data.py")
    write_json(
        META / "protected-sha256.json",
        {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected) if p.is_file()},
    )
    write_json(META / "sources.json", sources)
    write_json(
        META / "protocol.json",
        {
            "version": 1,
            "model": "YOLO11s Pose official COCO pretrained, no training",
            "weights_sha256": sha(WEIGHTS),
            "model_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt",
            "documentation": "https://docs.ultralytics.com/tasks/pose/",
            "device": 0,
            "imgsz": 1280,
            "target_fps": 15,
            "person_confidence": 0.25,
            "wrist_confidence": 0.35,
            "joint_confidence": 0.35,
            "counter_zone": camera["zones"]["counter"],
            "output_core": [
                [103 / 464, 446 / 848],
                [114 / 464, 443 / 848],
                [121 / 464, 474 / 848],
                [107 / 464, 479 / 848],
            ],
            "counter_output_zone": [
                [99 / 464, 438 / 848],
                [129 / 464, 438 / 848],
                [137 / 464, 479 / 848],
                [104 / 464, 489 / 848],
            ],
            "zone_provenance": (
                "Fixed from empty static X4 first frame apparatus geometry before any pose "
                "inference. Output zone includes wrist/palm allowance around visible lower mouth; "
                "not proof of physical contact. Same camera only."
            ),
            "reference_dimensions": [464, 848],
            "role_cache_max_age_sec": 0.35,
            "pose_role_min_iou": 0.30,
            "pose_role_margin": 0.15,
            "owner_min_samples": 3,
            "owner_min_span_sec": 0.12,
            "hand_track_max_gap_sec": 0.35,
            "hand_track_max_step_widths": 0.80,
            "hand_track_ambiguity_widths": 0.10,
            "approach_history_sec": 1.5,
            "approach_delta_widths": 0.20,
            "contact_min_samples": 2,
            "contact_min_sec": 0.05,
            "leave_distance_widths": 0.18,
            "leave_max_gap_sec": 0.35,
            "cash_followup_sec": 2.0,
            "cash_min_confidence": 0.25,
            "cash_min_samples": 2,
            "cash_wrist_distance_widths": 0.60,
            "cash_existing_before_sec": 0.5,
            "cash_new_min_gap_sec": 0.35,
            "duplicate_contact_sec": 0.40,
            "confidence_semantics": (
                "Heuristic evidence score, not calibrated probability. "
                "Wrist model confidence is retained separately."
            ),
            "decision_inputs": (
                "Relative time, skeleton arms/torso, cached geometric person roles, cash "
                "observations, camera polygons. No names, action annotations or known times."
            ),
            "face_policy": (
                "Facial keypoints discarded immediately; no recognition or identity embeddings."
            ),
            "no_training": True,
            "production_integration": False,
        },
    )
    print("Protocol, provenance and protected hashes frozen", flush=True)


def infer():
    import torch

    from app.cv.train_cash import local_yolo

    p = read_json(META / "protocol.json")
    assert sha(WEIGHTS) == p["weights_sha256"]
    assert torch.cuda.is_available(), "CUDA required for this diagnostic"
    model = local_yolo()(str(WEIGHTS))
    for s in read_json(META / "sources.json"):
        dest = DATA / s["id"] / "poses.json"
        if dest.exists():
            print("Cached", s["id"], flush=True)
            continue
        assert sha(s["video"]) == s["video_sha256"]
        cap = cv2.VideoCapture(s["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, round(fps / p["target_fps"]))
        frames, index, started = [], 0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                result = model.predict(
                    frame, imgsz=p["imgsz"], conf=p["person_confidence"], device=0, verbose=False
                )[0]
                people = []
                for box, conf, kpts in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.keypoints.data.cpu().tolist(),
                    strict=True,
                ):
                    # Only shoulders, elbows, wrists, hips. No facial landmarks persisted.
                    people.append(
                        {
                            "bbox_px": box,
                            "confidence": conf,
                            "joints": {str(k): kpts[k] for k in range(5, 13)},
                        }
                    )
                frames.append(
                    {
                        "frame_index": index,
                        "timestamp_sec": index / fps,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "poses": people,
                    }
                )
                if len(frames) % 150 == 0:
                    print(
                        s["id"],
                        len(frames),
                        "frames",
                        round(time.perf_counter() - started, 1),
                        "s",
                        flush=True,
                    )
            index += 1
        cap.release()
        write_json(
            dest,
            {
                "source_video_sha256": s["video_sha256"],
                "protocol_sha256": sha(META / "protocol.json"),
                "model_sha256": sha(WEIGHTS),
                "device": torch.cuda.get_device_name(0),
                "fps": fps,
                "stride": stride,
                "duration_sec": index / fps,
                "elapsed_sec": time.perf_counter() - started,
                "frames": frames,
            },
        )
        print("Saved", s["id"], len(frames), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "infer"])
    args = parser.parse_args()
    {"prepare": prepare, "infer": infer}[args.command]()
