"""Cached-pose R04 follow-up. predict never reads evaluation intervals or labels."""

import argparse
import json
import math
import time
from collections import Counter

import cv2
import numpy as np

from experiments.cash_actions import (
    analyze,
    apparatus_proposals,
    grade,
    handover_proposals,
    roi_evidence,
    wrist_cash,
)
from experiments.counter_occupancy import ROOT, sha
from experiments.counter_occupancy import read_json as read
from experiments.counter_occupancy import write_json as write
from experiments.hand_object_continuity import (
    PROTOCOL,
    follow_texture,
    palm_frames,
    seed_texture,
    template_similarity,
)
from experiments.run_cash_actions import evidence, roi_frames

OUT = ROOT / "data/hand_object_continuity_v1"
META = ROOT / "datasets/cash_actions_v1"


def pixel_tracks(video, cash, associations, dimensions, fps):
    """Seed from real cash detections, track at native resolution for at most 1 s."""
    owners = {(x["frame_index"], x["logical_cash_id"]): x for x in associations}
    seeds = {}
    w, h = dimensions
    for frame in cash:
        for obj in frame["objects"]:
            key = (frame["frame_index"], obj["logical_object_id"])
            if obj["label"] != "cash" or key not in owners or owners[key]["owner"] != "cashier":
                continue
            box = np.asarray(obj["bbox"]) * [w, h, w, h]
            seeds.setdefault(key[0], []).append((key[1], box, owners[key]))
    cap = cv2.VideoCapture(str(video))
    active, completed, rejected = [], [], []
    previous, last_index = None, None
    index = 0
    try:
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            if pixels.shape[:2] != (h, w):
                raise ValueError("Uncalibrated video dimensions")
            if index % 2 and index not in seeds:
                index += 1
                continue
            gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
            live = []
            for item in active:
                reason = None
                if (index - item["seed_frame"]) / fps > PROTOCOL["flow_max_seconds"]:
                    reason = "bounded_window_expired"
                elif (index - last_index) / fps > PROTOCOL["flow_max_frame_gap_sec"]:
                    reason = "frame_gap"
                if reason is None:
                    pts, shift, reason = follow_texture(previous, gray, item["points"])
                    if (
                        reason is None
                        and len(pts) / item["initial_points"] < PROTOCOL["flow_min_survival"]
                    ):
                        reason = "too_many_seed_features_lost"
                    if reason is None:
                        bbox = item["bbox"] + np.tile(shift, 2)
                        sim = template_similarity(gray, bbox, item["template"])
                        if sim is None or sim < PROTOCOL["flow_min_template_correlation"]:
                            reason = "appearance_changed_or_occluded"
                    if reason is None:
                        item.update(points=pts, bbox=bbox)
                        item["samples"].append(
                            {
                                "timestamp_sec": index / fps,
                                "frame_index": index,
                                "center_px": ((bbox[:2] + bbox[2:]) / 2).tolist(),
                                "bbox_px": bbox.tolist(),
                                "surviving_points": len(pts),
                                "template_correlation": sim,
                                "source": "optical_flow_texture_not_cash_detection",
                            }
                        )
                if reason is not None:
                    item["stop_reason"] = reason
                    completed.append(item)
                else:
                    live.append(item)
            active = live
            for identity, box, owner in seeds.get(index, []):
                pts, reason = seed_texture(gray, box)
                if reason:
                    rejected.append({"frame_index": index, "cash_id": identity, "reason": reason})
                    continue
                x1, y1, x2, y2 = np.rint(box).astype(int)
                active.append(
                    {
                        "seed_frame": index,
                        "seed_time": index / fps,
                        "cash_id": identity,
                        "seed_owner": owner,
                        "bbox": box,
                        "points": pts,
                        "initial_points": len(pts),
                        "template": gray[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)].copy(),
                        "samples": [],
                    }
                )
            previous, last_index = gray, index
            index += 1
    finally:
        cap.release()
    for item in active:
        item["stop_reason"] = "video_ended"
        completed.append(item)
    for item in completed:
        for key in ["bbox", "points", "template"]:
            del item[key]
        item["status"] = "texture_track_only"
        item["uncertainty"] = [
            "Features inside cash bbox may belong to the hand or background.",
            "Appearance continuity does not prove physical ownership or banknote identity.",
        ]
    return {"tracks": completed, "rejected_seeds": rejected}


def flow_ownership(tracks, hands, p):
    """Additional texture ownership cue; never emit cash/semantic observations."""
    for item in tracks:
        measurements = []
        seen = set()
        for node in item["samples"]:
            if not hands:
                break
            hf = min(hands, key=lambda f: abs(f["timestamp_sec"] - node["timestamp_sec"]))
            if abs(hf["timestamp_sec"] - node["timestamp_sec"]) > 0.04 or hf["frame_index"] in seen:
                continue
            seen.add(hf["frame_index"])
            ranked = sorted(
                [
                    (math.dist(node["center_px"], h["xy"]) / hf["counter_width_px"], i, h)
                    for i, h in enumerate(hf["hands"])
                ],
                key=lambda x: x[0],
            )
            if not ranked or ranked[0][0] > p["cash_wrist_distance_widths"]:
                continue
            distance, _, hand = ranked[0]
            competing = [x[2] for x in ranked if x[0] - distance < p["cash_owner_margin_widths"]]
            people = {(h["owner"], h.get("body_track_id")) for h in competing}
            owner = hand["owner"] if len(people) == 1 and hand.get("body_track_id") else "unknown"
            measurements.append(
                {
                    **node,
                    "owner": owner,
                    "hand_track_id": hand["track_id"],
                    "body_track_id": hand.get("body_track_id"),
                    "hand_xy": hand["xy"],
                    "hand_distance_widths": distance,
                    "pose_timestamp_sec": hf["timestamp_sec"],
                }
            )
        item["ownership_cues"] = measurements
        runs = []
        interrupted = True
        for node in measurements:
            if node["owner"] != "customer":
                interrupted = True
                continue
            if (
                interrupted
                or not runs
                or node["timestamp_sec"] - runs[-1][-1]["timestamp_sec"] > p["hand_gap_sec"]
                or node["hand_track_id"] != runs[-1][-1]["hand_track_id"]
            ):
                runs.append([])
            runs[-1].append(node)
            interrupted = False
        support = []
        for run in runs:
            if len(run) < 2 or run[-1]["timestamp_sec"] - run[0]["timestamp_sec"] < 0.1:
                continue
            object_motion = np.subtract(run[-1]["center_px"], run[0]["center_px"])
            hand_motion = np.subtract(run[-1]["hand_xy"], run[0]["hand_xy"])
            movement = np.linalg.norm(object_motion)
            residual = np.linalg.norm(object_motion - hand_motion)
            if movement >= 3 and residual <= 0.5 * movement:
                support.append(
                    {
                        "start": run[0]["timestamp_sec"],
                        "end": run[-1]["timestamp_sec"],
                        "movement_px": float(movement),
                        "motion_residual_px": float(residual),
                    }
                )
        if item.get("seed_owner", {}).get("owner") != "cashier":
            support = []
        item["recipient_motion_support"] = support
        item["status"] = "handover_texture_hypothesis" if support else "insufficient_evidence"
        if not support:
            item["uncertainty"].append("No sustained recipient-associated co-motion after seed.")


def predict():
    if (OUT / "summary.json").exists():
        raise FileExistsError(
            "Preserve the first run; choose a new experiment output for revisions"
        )
    write(OUT / "protocol.json", PROTOCOL)
    protected = list((ROOT / "app").rglob("*.py")) + list((ROOT / "config").rglob("*.json"))
    protected += [
        ROOT / "data/cash/models/cash.pt",
        ROOT / "data/counter_hand_v1/pretrained/yolo11s-pose.pt",
    ]
    hashes = {str(f.relative_to(ROOT)): sha(f) for f in protected}
    write(OUT / "protected-before.json", hashes)
    p = read(META / "protocol.json")
    summary = []
    for source in read(META / "sources.json"):
        started = time.perf_counter()
        video = ROOT / source["video"]
        poses = read(ROOT / "data/counter_hand_v1" / source["id"] / "poses.json")
        hands = read(ROOT / "data/counter_hand_v1" / source["id"] / "hands.json")["frames"]
        cache = read(ROOT / source["cache"] / "detections.json")
        assert (
            sha(video)
            == poses["source_video_sha256"]
            == cache["video_sha256"]
            == source["video_sha256"]
        )
        tracks = read(ROOT / source["cache"] / "track_history.json")["frames"]
        assert len(hands) == len(poses["frames"])
        assert all(
            h["frame_index"] == f["frame_index"]
            and abs(h["timestamp_sec"] - f["timestamp_sec"]) < 0.001
            for h, f in zip(hands, poses["frames"], strict=True)
        )
        assert all(
            abs(t["timestamp_sec"] - f["timestamp_sec"]) < 0.001
            for t, f in zip(tracks, cache["frames"], strict=True)
        )
        cash = [
            {**t, "frame_index": f["frame_index"]}
            for t, f in zip(tracks, cache["frames"], strict=True)
        ]
        palms = palm_frames(hands)
        rois = roi_frames(video, hands, p)  # physical masks stay based on observed wrists
        result = analyze(palms, cash, rois, p, poses["duration_sec"])
        for candidate in result["candidates"]:
            candidate["position_basis"] = "projected_palm_not_measured_keypoint"
            candidate["uncertainty"].append(
                "Unvalidated straight-hand geometry; flexed fingers/palm may differ."
            )
        output = OUT / source["id"]
        evidence(video, result["candidates"], output, poses["fps"], p)
        write(output / "actions.json", result)
        baseline = wrist_cash(hands, cash, p)
        original = {(n["frame_index"], n["logical_cash_id"]): n for n in baseline}
        changed = [
            {"before": original[(n["frame_index"], n["logical_cash_id"])], "after": n}
            for n in result["cash_wrist_observations"]
            if (n["frame_index"], n["logical_cash_id"]) in original
            and original[(n["frame_index"], n["logical_cash_id"])]["owner"] != n["owner"]
        ]
        write(output / "association-changes.json", changed)
        flow = pixel_tracks(
            video, cash, result["cash_wrist_observations"], p["dimensions"], poses["fps"]
        )
        flow_ownership(flow["tracks"], palms, p)
        write(output / "texture-tracks.json", flow)
        row = {
            "video": source["id"],
            "elapsed_sec": time.perf_counter() - started,
            "action_hypotheses": dict(
                Counter(c["type"] for c in result["candidates"] if c["status"] == "hypothesis")
            ),
            "association_changes": len(changed),
            "flow_seeds": len(flow["tracks"]),
            "flow_rejected_seeds": len(flow["rejected_seeds"]),
            "flow_handover_hypotheses": sum(
                bool(t["recipient_motion_support"]) for t in flow["tracks"]
            ),
            "confirmed_events": 0,
        }
        summary.append(row)
        print(json.dumps(row), flush=True)
    assert all(sha(ROOT / path) == value for path, value in hashes.items())
    write(
        OUT / "summary.json",
        {
            "videos": summary,
            "protected_files_unchanged": len(hashes),
            "production_integrated": False,
        },
    )


def score():
    """Only evaluation opens human review labels; no influence on predictions."""
    labels = read(META / "evaluation-only.json")
    negatives = read(META / "visual-review.json")["negative_reviews"]
    rows = []
    for source in read(META / "sources.json"):
        name = source["id"]
        targets = labels["videos"][name]["actions"]
        variants = [
            ("wrist", ROOT / "data/cash_actions_v1", "actions.json"),
            ("palm", OUT, "actions.json"),
        ]
        if (OUT / name / "association-only.json").exists():
            variants.append(("association_only", OUT, "association-only.json"))
        for variant, folder, filename in variants:
            candidates = read(folder / name / filename)["candidates"]
            strong = [c for c in candidates if c["status"] == "hypothesis"]
            matched = set()
            scored = []
            for t in targets:
                hits = [
                    c
                    for c in strong
                    if c["type"] == t["type"]
                    and t["start_sec"] <= c["timestamp_sec"] <= t["end_sec"]
                ]
                matched.update(c["id"] for c in hits)
                scored.append({**t, "hits": [c["id"] for c in hits]})
            rows.append(
                {
                    "video": name,
                    "variant": variant,
                    "targets": scored,
                    "unmatched_hypotheses": [c["id"] for c in strong if c["id"] not in matched],
                    "negative_hits": [
                        {
                            **n,
                            "hits": [
                                c["id"]
                                for c in strong
                                if n["start_sec"] <= c["timestamp_sec"] <= n["end_sec"]
                            ],
                        }
                        for n in negatives
                        if n["video"] == name
                    ],
                }
            )
    write(
        OUT / "evaluation.json",
        {
            "rows": rows,
            "limitations": labels["limitations"],
            "unmatched_are_not_automatically_false": True,
        },
    )
    print(json.dumps(rows, indent=2))


def association_only():
    """Post-first-run ablation: observed wrist contacts + alternative cash association.

    No thresholds are changed and no texture samples become detections.
    This remains diagnostic development on the same videos, not independent validation.
    """
    p = read(META / "protocol.json")
    for source in read(META / "sources.json"):
        output = OUT / source["id"]
        if (output / "association-only.json").exists():
            raise FileExistsError("Preserve the ablation results")
        hands = read(ROOT / "data/counter_hand_v1" / source["id"] / "hands.json")["frames"]
        poses = read(ROOT / "data/counter_hand_v1" / source["id"] / "poses.json")
        alternative = read(output / "actions.json")["cash_wrist_observations"]
        rois = roi_frames(source["video"], hands, p)
        candidates = apparatus_proposals(hands, alternative, p, poses["duration_sec"])
        candidates += handover_proposals(hands, alternative, p, poses["duration_sec"])
        candidates = [c for c in candidates if c["window_end_sec"] - c["window_start_sec"] >= 1]
        candidates.sort(key=lambda c: (c["timestamp_sec"], c["type"]))
        for i, candidate in enumerate(candidates):
            candidate["id"] = f"association-{i + 1}"
            candidate["observations"]["hands"] = [
                h
                for h in candidate["observations"]["hands"]
                if candidate["window_start_sec"]
                <= h["timestamp_sec"]
                <= candidate["window_end_sec"]
            ]
            candidate["contact_end_sec"] = min(
                candidate["contact_end_sec"], candidate["window_end_sec"]
            )
            candidate["features"]["contact_duration_sec"] = (
                candidate["contact_end_sec"] - candidate["timestamp_sec"]
            )
            candidate["roi"] = roi_evidence(candidate, rois, p)
            grade(candidate, p)
            candidate["uncertainty"].append(
                "Cash ownership uses unvalidated projected palm geometry."
            )
            candidate["evidence_frames"] = []
        evidence(source["video"], candidates, output, poses["fps"], p)
        write(
            output / "association-only.json",
            {"candidates": candidates, "confirmed_events": [], "production_integration": False},
        )
        print(
            source["id"],
            [(c["type"], c["timestamp_sec"]) for c in candidates if c["status"] == "hypothesis"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["predict", "score", "association_only"])
    globals()[parser.parse_args().command]()
