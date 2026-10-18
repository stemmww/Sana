"""Offline diagnostics from existing YOLO/pose caches and original video pixels."""

import argparse
import html
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from experiments.cash_actions import analyze
from experiments.counter_occupancy import ROOT, imread, imwrite, sha
from experiments.counter_occupancy import read_json as read
from experiments.counter_occupancy import write_json as write

DATA = ROOT / "data/cash_actions_v1"
META = ROOT / "datasets/cash_actions_v1"


def prepare():
    prior = read(ROOT / "datasets/counter_hand_v1/protocol.json")
    counter = prior["counter_zone"]
    ymin = min(y for x, y in counter)
    ymax = max(y for x, y in counter)
    cutoff = ymin + 0.45 * (ymax - ymin)
    # Polygon clipping against a fixed horizontal tray boundary, not frame labels.
    upper = []
    for a, b in zip(counter, counter[1:] + counter[:1]):
        if a[1] <= cutoff:
            upper.append(a)
        if (a[1] <= cutoff) != (b[1] <= cutoff):
            alpha = (cutoff - a[1]) / (b[1] - a[1])
            upper.append([a[0] + alpha * (b[0] - a[0]), cutoff])
    protocol = {
        "version": 1,
        "zones": {"counter": counter, "input": upper, "output": prior["counter_output_zone"]},
        "dimensions": prior["reference_dimensions"],
        "input_zone_provenance": "upper 45% of existing counter polygon, static geometry only",
        "window_sec": 3.0,
        "window_before_sec": 1.0,
        "hand_gap_sec": prior["hand_track_max_gap_sec"],
        "approach_delta_widths": 0.2,
        "leave_delta_widths": 0.18,
        "input_wrist_allowance_widths": 0.18,
        "cash_pose_time_tolerance_sec": 0.10,
        "cash_confidence": 0.25,
        "cash_wrist_distance_widths": 0.6,
        "cash_owner_margin_widths": 0.1,
        "cash_min_samples": 2,
        "handover_contact_widths": 0.6,
        "handover_leave_widths": 0.8,
        "roi_min_unmasked_fraction": 0.4,
        "roi_pixel_delta": 20.0,
        "roi_change_fraction": 0.05,
        "hand_mask_radius_widths": 0.30,
        "confirmed_event_gate": "independent validation required",
        "no_training": True,
        "new_model_inference": False,
        "production_integration": False,
    }
    write(META / "protocol.json", protocol)
    sources = [
        s
        for s in read(ROOT / "datasets/counter_hand_v1/sources.json")
        if s["id"] in {"x1", "x3", "x5"}
    ]
    write(META / "sources.json", sources)
    protected = list((ROOT / "app").rglob("*.py")) + list((ROOT / "config").rglob("*.json"))
    protected += [
        ROOT / "data/cash/models/cash.pt",
        ROOT / "data/counter_hand_v1/pretrained/yolo11s-pose.pt",
    ]
    write(META / "protected-before.json", {str(f.relative_to(ROOT)): sha(f) for f in protected})
    # Freeze prior evaluation labels separately. Predict never opens this file.
    old = read(ROOT / "datasets/counter_sequence_v1/ground-truth.json")
    targets = {}
    for s in sources:
        rows = []
        value = old["videos"][s["id"]]
        for act in value["actions"]:
            rows.append(
                {
                    "type": "CASH_LOADING_ACTION"
                    if act["id"] == "deposit"
                    else "CASH_EXTRACTION_ACTION",
                    "start_sec": act["start_sec"],
                    "end_sec": act["end_sec"],
                }
            )
        if "customer_return_interval" in value:
            a, b = value["customer_return_interval"]
            rows.append({"type": "CASH_HANDOVER_TO_CUSTOMER", "start_sec": a, "end_sec": b})
        targets[s["id"]] = {"actions": rows, "duration_note": value["note"]}
    write(
        META / "evaluation-only.json",
        {
            "provenance": old["provenance"],
            "limitations": [
                "Repeated development clips, not independent action validation",
                "No verified empty lower-tray touch video; user confirmed unavailable",
                "Unknown actions outside reviewed targets require visual review before counting as false",
            ],
            "videos": targets,
        },
    )
    pixels = imread(ROOT / "data/counter_hand_v1/calibration-raw.jpg")
    for name, poly in protocol["zones"].items():
        xy = np.rint(np.array(poly) * protocol["dimensions"]).astype(np.int32)
        cv2.polylines(pixels, [xy], True, (0, 255, 255) if name == "input" else (255, 100, 0), 1)
        cv2.putText(pixels, name, tuple(xy[0]), 0, 0.4, (0, 0, 255), 1)
    imwrite(DATA / "zones.jpg", pixels)


def roi_frames(video, hands, p):
    indices = {f["frame_index"]: f for f in hands}
    geometries = {}
    w, h = p["dimensions"]
    for zone in ["input", "output"]:
        points = np.rint(np.array(p["zones"][zone]) * [w, h]).astype(np.int32)
        x, y, cw, ch = cv2.boundingRect(points)
        mask = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(mask, [points - [x, y]], 1)
        geometries[zone] = (x, y, cw, ch, mask.astype(bool))
    result = []
    cap = cv2.VideoCapture(str(video))
    i = 0
    try:
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            if i in indices:
                if pixels.shape[:2] != (h, w):
                    raise ValueError("Uncalibrated camera dimensions")
                f = indices[i]
                row = {"frame_index": i, "timestamp_sec": f["timestamp_sec"]}
                gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
                for zone, (x, y, cw, ch, mask) in geometries.items():
                    handmask = np.zeros((ch, cw), np.uint8)
                    for hand in f["hands"]:
                        center = np.rint(np.array(hand["xy"]) - [x, y]).astype(int)
                        cv2.circle(
                            handmask,
                            tuple(center),
                            round(f["counter_width_px"] * p["hand_mask_radius_widths"]),
                            1,
                            -1,
                        )
                    row[zone] = {
                        "gray": gray[y : y + ch, x : x + cw].copy(),
                        "mask": mask,
                        "hands": handmask.astype(bool),
                    }
                result.append(row)
            i += 1
    finally:
        cap.release()
    if len(result) != len(indices):
        raise ValueError("Missing ROI frames")
    return result


def evidence(video, candidates, output, fps, p):
    cap = cv2.VideoCapture(str(video))
    for c in candidates:
        chosen = set(
            round(t * fps) for t in np.linspace(c["window_start_sec"], c["window_end_sec"], 6)
        )
        chosen.update(
            n["frame_index"]
            for n in c["observations"]["cash_before"] + c["observations"]["cash_after"]
        )
        panels = []
        for index in sorted(chosen):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, img = cap.read()
            if not ok:
                continue
            path = output / "evidence" / f"{index:05d}.jpg"
            imwrite(path, img)
            c["evidence_frames"].append(
                {
                    "frame_index": index,
                    "timestamp_sec": index / fps,
                    "path": str(path.relative_to(ROOT)),
                }
            )
            canvas = img[275:535, :360].copy()
            cv2.putText(canvas, f"{index / fps:.3f}s {c['id']}", (3, 16), 0, 0.45, (0, 0, 255), 1)
            panels.append(canvas)
        while panels and len(panels) % 3:
            panels.append(np.zeros_like(panels[0]))
        if panels:
            imwrite(
                output / f"{c['id']}.jpg",
                np.vstack([np.hstack(panels[i : i + 3]) for i in range(0, len(panels), 3)]),
            )
    cap.release()


def predict():
    p = read(META / "protocol.json")
    for source in read(META / "sources.json"):
        video = Path(source["video"])
        out = DATA / source["id"]
        base = ROOT / "data/counter_hand_v1" / source["id"]
        poses = read(base / "poses.json")
        cache = read(ROOT / source["cache"] / "detections.json")
        if (
            sha(video) != source["video_sha256"]
            or poses["source_video_sha256"] != source["video_sha256"]
            or cache["video_sha256"] != source["video_sha256"]
        ):
            raise ValueError("Source/cache SHA mismatch")
        tracks = read(ROOT / source["cache"] / "track_history.json")["frames"]
        hands = read(base / "hands.json")["frames"]
        cash = [
            {**f, "frame_index": r["frame_index"]}
            for f, r in zip(tracks, cache["frames"], strict=True)
        ]
        started = time.perf_counter()
        rois = roi_frames(video, hands, p)
        result = analyze(hands, cash, rois, p, poses["duration_sec"])
        result["elapsed_analysis_sec"] = time.perf_counter() - started
        evidence(video, result["candidates"], out, poses["fps"], p)
        result.update(
            source_sha256=source["video_sha256"],
            protocol_sha256=sha(META / "protocol.json"),
            decision_code_sha256=sha(ROOT / "experiments/cash_actions.py"),
            cash_cache_sha256=sha(ROOT / source["cache"] / "detections.json"),
            hands_sha256=sha(base / "hands.json"),
            pose_sha256=sha(base / "poses.json"),
            elapsed_with_evidence_sec=time.perf_counter() - started,
            cached_inference=True,
            new_models_run=False,
        )
        write(out / "actions.json", result)
        (out / "timeline.txt").write_text(
            "\n".join(
                f"{c['timestamp_sec']:.3f} {c['type']} "
                f"{c['status']} {c['confidence']} {c['uncertainty']}"
                for c in result["candidates"]
            ),
            encoding="utf-8",
        )
        print(
            source["id"],
            Counter(c["type"] for c in result["candidates"]),
            Counter(c["type"] for c in result["candidates"] if c["status"] == "hypothesis"),
            flush=True,
        )


def score():
    gt = read(META / "evaluation-only.json")
    rows = []
    document = """<!doctype html><meta charset="utf-8"><title>Sana cash actions</title>
<style>body{font:16px/1.5 system-ui;margin:30px;max-width:1300px}td,th{border:1px solid #aaa;
padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap}img{max-width:100%}</style>
<h1>Action-based R04 diagnostic — no production integration</h1>
<p>Only saved cash/pose/wrist observations and OpenCV ROI differences. Windows ≤3s.
Scores are not probabilities. Hypotheses are NOT business events or confirmed violations.</p>"""
    for source in read(META / "sources.json"):
        name = source["id"]
        result = read(DATA / name / "actions.json")
        targets = gt["videos"][name]["actions"]
        matched = set()
        comparisons = []
        for c in result["candidates"]:
            matches = [
                i
                for i, a in enumerate(targets)
                if a["type"] == c["type"]
                and a["start_sec"] <= c["contact_end_sec"]
                and c["timestamp_sec"] <= a["end_sec"]
            ]
            if c["status"] == "hypothesis":
                available = [i for i in matches if i not in matched]
                if available:
                    matched.add(available[0])
            comparisons.append(
                {
                    "candidate_id": c["id"],
                    "matched_target_indices": matches,
                    "status": c["status"],
                    "needs_visual_review": not bool(matches),
                }
            )
        counts = {}
        for kind in ["CASH_LOADING_ACTION", "CASH_EXTRACTION_ACTION", "CASH_HANDOVER_TO_CUSTOMER"]:
            indices = [i for i, a in enumerate(targets) if a["type"] == kind]
            strong = [
                c for c in result["candidates"] if c["type"] == kind and c["status"] == "hypothesis"
            ]
            hit = len(set(indices) & matched)
            counts[kind] = {
                "real_actions": len(indices),
                "hypothesis_hits": hit,
                "missed": len(indices) - hit,
                "unmatched_or_duplicate_hypotheses": len(strong) - hit,
                "confirmed": 0,
            }
        row = {
            "video": name,
            "counts": counts,
            "candidates": comparisons,
            "analysis_sec": result["elapsed_analysis_sec"],
            "with_evidence_sec": result["elapsed_with_evidence_sec"],
        }
        rows.append(row)
        document += f"<h2>{name.upper()}</h2><pre>{html.escape(json.dumps(counts, indent=2))}</pre>"
        document += f'<a href="{name}/actions.json">All observations</a> | <a href="{name}/timeline.txt">Timeline</a>'
        for c in result["candidates"]:
            document += f"<details><summary>{c['id']} {c['timestamp_sec']:.3f}s {c['type']} {c['status']}</summary>"
            document += f'<img src="{name}/{c["id"]}.jpg"><pre>{html.escape(json.dumps(c, indent=2))}</pre></details>'
    changed = [f for f, v in read(META / "protected-before.json").items() if sha(ROOT / f) != v]
    write(
        DATA / "summary.json",
        {
            "videos": rows,
            "limitations": gt["limitations"],
            "protected_changes": changed,
            "confirmed_events": 0,
            "production_integrated": False,
        },
    )
    document += "<h2>Limitations</h2><pre>" + html.escape(str(gt["limitations"])) + "</pre>"
    (DATA / "report.html").write_text(document, encoding="utf-8")
    if changed:
        raise AssertionError(changed)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "predict", "score"])
    globals()[parser.parse_args().command]()
