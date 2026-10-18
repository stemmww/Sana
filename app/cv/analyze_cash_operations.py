"""Opt-in CashOperation analysis using existing local detector/pose caches.

Select --profile explicitly after inspecting its reference image. Geometry is
verified throughout the video before any zones are applied. R01–R05 are unchanged;
the new operation events are a separate output, never synthetic customer events.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.cv.analysis_result import assemble_video_result
from app.cv.analyze_events import analyze_video, sha256
from app.cv.camera_profile import CameraProfile
from app.events.cash_operation_observations import cash_observations
from app.events.cash_operations import CashOperationDetector
from app.services.analysis_result import compose_analysis_result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def image(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", pixels)
    if not ok:
        raise OSError("Cannot encode evidence image")
    encoded.tofile(path)


def run(video, profile_path, replay, poses_path, output):
    profile = CameraProfile(profile_path)
    cache, poses = read(replay), read(poses_path)
    digest = sha256(video)
    if cache["video_sha256"] != digest or poses["source_video_sha256"] != digest:
        raise ValueError("Observation cache belongs to a different video")
    cash_weights = Path(__file__).resolve().parents[2] / "data/cash/models/cash.pt"
    if cache["cash_weights_sha256"] != sha256(cash_weights):
        raise ValueError("Cached cash model differs from the current production model")
    output.mkdir(parents=True, exist_ok=True)
    indices = {f["frame_index"] for f in cache["frames"]}
    cap = cv2.VideoCapture(str(video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    checks, index = [], 0
    while True:
        ok, pixels = cap.read()
        if not ok:
            break
        if index in indices:
            check = profile.validate(pixels)
            checks.append({"timestamp": index / fps, "frame_index": index, **check})
            image(output / "evidence" / f"{index:05d}.jpg", pixels)
            if len(checks) == 1:
                image(output / "zones.jpg", profile.overlay(pixels))
        index += 1
    cap.release()
    valid = bool(checks) and all(c["valid"] for c in checks) and len(checks) == len(indices)
    write(
        output / "camera-validation.json",
        {
            "profile_id": profile.data["profile_id"],
            "selection": "explicit_argument",
            "valid": valid,
            "checks": checks,
        },
    )
    if not valid:
        result = {
            "status": "insufficient_evidence",
            "operations": [],
            "events": [],
            "reason": "camera_geometry_unverified",
            "profile": profile.data["profile_id"],
        }
        write(output / "cash-operations.json", result)
        (output / "timeline.txt").write_text(
            "Camera rejected; no operation events", encoding="utf-8"
        )
        api_result = compose_analysis_result({}, result).model_dump(mode="json")
        write(output / "api-result.json", api_result)
        return api_result
    config_path = output / "event-config.json"
    write(config_path, profile.data["event_config"])
    # Same existing person tracker, logical cash tracker, sessions and business rules.
    analyze_video(video, output / "legacy", config_path, replay=replay)
    frames = read(output / "legacy/track_history.json")["frames"]
    detector, pi, evidence = CashOperationDetector(), -1, []
    for frame, raw in zip(frames, cache["frames"], strict=True):
        timestamp = frame["timestamp_sec"]
        while (
            pi + 1 < len(poses["frames"]) and poses["frames"][pi + 1]["timestamp_sec"] <= timestamp
        ):
            pi += 1
        path = str((output / "evidence" / f"{raw['frame_index']:05d}.jpg").resolve())
        observations = cash_observations(
            frame,
            poses["frames"][pi] if pi >= 0 else None,
            profile.data["zones"],
            raw["width"],
            raw["height"],
            path,
        )
        detector.update(timestamp, observations)
        evidence.append({"timestamp": timestamp, "cash": [o.model_dump() for o in observations]})
    result = detector.finish(count / fps)
    result.update(profile=profile.data["profile_id"], video_sha256=digest)
    write(output / "cash-observations.json", evidence)
    write(output / "cash-operations.json", result)
    timeline = [f"PROFILE {profile.data['profile_id']} geometry verified"]
    for operation in result["operations"]:
        timeline.append(
            f"{operation['start_time']:.3f} START {operation['operation_id']} "
            f"session={operation['associated_session_id']} cash={operation['logical_cash_ids']}"
        )
        timeline += [
            f"{e['timestamp']:.3f} {e['type']} {e['status']} confidence={e['confidence']:.3f}"
            for e in operation["events"]
        ]
        timeline.append(
            f"END {operation['end_time']} {operation['completion_status']} "
            f"uncertainty={operation['uncertainty']}"
        )
    if not result["operations"]:
        timeline.append("No sufficiently supported moving cash operation")
    (output / "timeline.txt").write_text("\n".join(timeline), encoding="utf-8")
    render_debug(video, profile, cache, evidence, result, output)
    api_result = assemble_video_result(video, read(output / "legacy/events.json"), result)
    write(output / "api-result.json", api_result.model_dump(mode="json"))
    return api_result.model_dump(mode="json")


def render_debug(video, profile, cache, evidence, result, output):
    by_frame = {raw["frame_index"]: obs for raw, obs in zip(cache["frames"], evidence, strict=True)}
    events = sorted(
        [e for op in result["operations"] for e in op["events"]], key=lambda e: e["timestamp"]
    )
    cap = cv2.VideoCapture(str(video))
    writer, index = None, 0
    try:
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            if index in by_frame:
                canvas = profile.overlay(pixels)
                obs = by_frame[index]
                for cash in obs["cash"]:
                    x, y = np.rint(np.array(cash["center"]) * [pixels.shape[1], pixels.shape[0]])
                    cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 255), 1)
                    cv2.putText(
                        canvas,
                        cash["logical_cash_id"] + ":" + cash["location"],
                        (int(x), int(y) - 6),
                        0,
                        0.35,
                        (0, 255, 255),
                        1,
                    )
                recent = [e for e in events if e["timestamp"] <= obs["timestamp"]][-3:]
                cv2.rectangle(canvas, (0, 0), (pixels.shape[1], 95), (20, 20, 20), -1)
                lines = [f"{obs['timestamp']:.2f}s {profile.data['profile_id']} CashOperation"]
                lines += [f"{e['type']} {e['status']}" for e in recent]
                for n, line in enumerate(lines):
                    cv2.putText(canvas, line, (5, 19 + n * 22), 0, 0.42, (255, 255, 255), 1)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output / "debug.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        cache["fps"] / cache["stride"],
                        (pixels.shape[1], pixels.shape[0]),
                    )
                    if not writer.isOpened():
                        raise OSError("Cannot create operation debug video")
                writer.write(canvas)
            index += 1
    finally:
        cap.release()
        if writer:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.video, args.profile, args.replay, args.poses, args.output)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "cash_operation_status": result["diagnostics"]["cash_operation_status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
