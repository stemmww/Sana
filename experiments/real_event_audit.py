"""Isolated one-variable cash inference experiment; never promotes weights or events.

Evaluation manifest selects sources/camera calibration, not decisions. Ground truth
is read only by review/scoring. Person and pose observations are frozen to isolate
cash inference resolution. All downstream production components stay unchanged.
"""

import argparse
import copy
import html
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.cv.analysis_result import assemble_video_result
from app.cv.analyze_cash_operations import read, run, write
from app.cv.analyze_events import analyze_video, sha256
from app.cv.evaluate_cash import iou
from app.cv.evaluate_objects import score_frames
from app.cv.inspect_cash import DEFAULT_MODEL
from app.cv.object_dataset import validated_items, verify_holdout
from app.cv.object_detector import ObjectDetector
from app.events.config import EventConfig
from app.events.detector import EventDetector
from app.events.observations import suppress_object_duplicates
from app.events.schemas import FrameObservation
from app.events.tracking import LogicalObjectTracker
from app.services.person_roles import point_in_polygon
from experiments.counter_occupancy import ROOT, imwrite

DATA = ROOT / "data/real_event_audit_v1"
MANIFEST = ROOT / "datasets/verdict_separation_v1/evaluation.json"
CANDIDATE = ROOT / "data/cash/runs/cash_id_v1_run2/weights/best.pt"


def cache_path(source):
    return ROOT / (
        source["detections"]
        if "detections" in source
        else source["observations"] + "/input-detections.json"
    )


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    paths = list((ROOT / "app").rglob("*.py")) + list((ROOT / "config").rglob("*.json"))
    paths += [ROOT / DEFAULT_MODEL, CANDIDATE]
    write(DATA / "protected-before.json", {str(p.relative_to(ROOT)): sha256(p) for p in paths})
    write(
        DATA / "protocol.json",
        {
            "single_changed_layer": "cash object inference input size 768 -> 1280",
            "weights": str(DEFAULT_MODEL),
            "confidence": 0.25,
            "device": 0,
            "person_and_pose": "frozen measured observations; inference time excluded",
            "all_video_frames": "same sampled indices as baseline, no action-time selection",
            "promotion": "never automatic; require real R04 and no added normal-video alarms",
            "guardrails": "no cash recall loss or cash FP increase on fixed 42 frames",
            "evaluation_limit": "repeated development validation, not independent test",
        },
    )
    rows = [
        [
            "R04",
            "MONEY_TO_COUNTER",
            "No cash observation in counter zone in X-1",
            "Stable cashier -> counter observations of same stack",
            "Sparse small-object detections; fragments before load",
            "Test existing cash weights at 1280, fixed .25 threshold",
            "Extra false cash / changed localization",
        ],
        [
            "R04",
            "MONEY_FROM_COUNTER",
            "One pose contact/leave hypothesis, followed by cash-10/11",
            "Cash positively linked to counter origin",
            "Wrist contact is ambiguous; no prior confirmed counter cash",
            "Do not promote pose alone; first repair observable counter evidence",
            "Empty touches mistaken for withdrawal",
        ],
        [
            "R04",
            "MONEY_HELD_BY_CASHIER -> MONEY_TRANSFER_TO_CUSTOMER",
            "Cash-11 moves table/cashier/customer; no stable transition",
            "Stable ownership and lineage after withdrawal",
            "Person-center association misses extended hand; short fragments",
            "After detection gate: evaluate wrist-backed ownership in separate experiment",
            "False transfer from hand proximity",
        ],
        [
            "R05",
            "Previous-session cash still on table at next customer",
            "cash-22 present at session 2; cash-20 -> 22 gap 2.338s",
            "Identity through occlusion and confirmed counter withdrawal",
            "IoU=0; appearance .730 is not identity; forward flow misses target",
            "Retain insufficient_evidence; collect disambiguating occlusion observations",
            "Merging a new pile or background false detection",
        ],
        [
            "R03",
            "ID_TRANSFER_TO_CASHIER",
            "Production cash model has no id_card class",
            "Reliable customer -> cashier ID trajectory",
            "ID channel intentionally disabled; candidate not validated",
            "Offline candidate inspection on X-2 and X-3 only; no promotion",
            "ID/cash confusion and missed transfers",
        ],
        [
            "R03",
            "ID_CHECK -> ID_ON_TABLE",
            "No production ID events; CHECK is a held-ID proxy",
            "Transfer, stable cashier hold, table transition; reliable absence coverage for R03",
            "No verified ID action coverage; holding ID does not establish visual inspection",
            "Report candidate temporal availability; keep coverage unknown",
            "Treating missed ID as skipped procedure",
        ],
    ]
    write(
        DATA / "audit-before.json",
        {
            "columns": [
                "Rule",
                "Expected event",
                "Actual observation",
                "Missing evidence",
                "Root cause",
                "Minimal fix",
                "Regression risk",
            ],
            "rows": rows,
        },
    )
    overlap = {}
    for name in ("cash_id_v1", "cash_id_v2"):
        items = validated_items(read(ROOT / f"datasets/{name}/annotations.json"))
        train = Counter(i["video_sha256"] for i in items if i["split"] == "train")
        val = Counter(i["video_sha256"] for i in items if i["split"] == "val")
        overlap[name] = {
            s["id"]: {
                "train_frames": train[s["video_sha256"]],
                "validation_frames": val[s["video_sha256"]],
            }
            for s in read(MANIFEST)
        }
    write(DATA / "training-overlap.json", overlap)


def infer(video, cache, detector):
    """Replace only object observations; no source names/times influence prediction."""
    if sha256(video) != cache["video_sha256"]:
        raise ValueError("Wrong source video")
    started = time.perf_counter()
    result = copy.deepcopy(cache)
    by_index = {r["frame_index"]: r for r in result["frames"]}
    capture, index, seen = cv2.VideoCapture(str(video)), 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in by_index:
                row = by_index[index]
                height, width = frame.shape[:2]
                if (width, height) != (row["width"], row["height"]):
                    raise ValueError("Observation geometry mismatch")
                row["objects"] = [
                    {
                        "label": d["label"],
                        "confidence": d["confidence"],
                        "bbox": [
                            v / (width if k % 2 == 0 else height)
                            for k, v in enumerate(d["box_xyxy"])
                        ],
                    }
                    for d in detector.detect(frame)
                ]
                seen += 1
            index += 1
    finally:
        capture.release()
    if seen != len(by_index):
        raise ValueError("Missing decoded frames")
    result["cash_weights_sha256"] = detector.model_sha256
    result["experiment"] = {
        "imgsz": detector.imgsz,
        "confidence": detector.confidence,
        "device": detector.device,
        "elapsed_sec": time.perf_counter() - started,
        "people_reused": True,
        "model_classes": sorted(detector.supported),
    }
    return result


def pipeline():
    detector = ObjectDetector(ROOT / DEFAULT_MODEL, 0.25, 1280, 0)
    for source in read(MANIFEST):
        output = DATA / source["id"]
        video = Path(source["video"])
        cache = infer(video, read(cache_path(source)), detector)
        write(output / "input-detections.json", cache)
        started = time.perf_counter()
        if source["mode"] == "cash_operation":
            api = run(
                video,
                ROOT / source["profile"],
                output / "input-detections.json",
                ROOT / source["observations"] / "poses.json",
                output,
            )
        else:
            analyze_video(
                video,
                output / "legacy",
                ROOT / source["config"],
                replay=output / "input-detections.json",
            )
            api = assemble_video_result(video, read(output / "legacy/events.json")).model_dump(
                mode="json"
            )
        # Do not attach old track-linked hypotheses to newly numbered cash identities.
        # Preserve the original diagnostics separately, with their provenance explicit.
        baseline = read(ROOT / "data/verdict_separation_v1" / source["id"] / "api-result.json")
        api["diagnostics"]["prior_observation_diagnostics"] = {
            "scope": "baseline only, old detections/track IDs; not current evidence",
            "diagnostics": baseline["diagnostics"],
        }
        write(output / "api-result.json", api)
        write(
            output / "timing.json",
            {
                "cash_inference_decode_sec": cache["experiment"]["elapsed_sec"],
                "downstream_pipeline_sec": time.perf_counter() - started,
                "excludes": "initial model load; cached person/pose inference",
            },
        )
        print(source["id"], api["verdict"], len(api["violations"]), flush=True)


def validation():
    annotations = read(ROOT / "datasets/cash_id_v1/annotations.json")
    reviewed = validated_items(annotations)
    verify_holdout(annotations, reviewed)
    items = [i for i in reviewed if i["split"] == "val"]
    results = {}
    for size in (768, 1280):
        detector = ObjectDetector(ROOT / DEFAULT_MODEL, 0.25, size, 0)
        frames = []
        for item in items:
            if sha256(ROOT / item["image"]) != item["image_sha256"]:
                raise ValueError("Validation image changed")
            pixels = cv2.imdecode(np.fromfile(ROOT / item["image"], np.uint8), cv2.IMREAD_COLOR)
            frames.append({**item, "detections": detector.detect(pixels)})
        results[str(size)] = {"metrics": score_frames(frames), "frames": frames}
    write(DATA / "resolution-validation.json", results)
    print({k: v["metrics"] for k, v in results.items()}, flush=True)


def counter_crop():
    """One frozen counter crop probe; only counter-center predictions are added.

    Uses calibration, never annotation boxes. No event logic or promotion here.
    """
    polygon = read(ROOT / "config/event_camera.json")["zones"]["counter"]
    points = np.array(polygon)
    lo, hi = points.min(axis=0), points.max(axis=0)
    lower, upper = np.maximum(0, lo - (hi - lo)), np.minimum(1, hi + (hi - lo))
    write(
        DATA / "counter-crop-protocol.json",
        {
            "polygon": polygon,
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "imgsz": 768,
            "confidence": 0.25,
            "nms_iou": 0.5,
            "gate": (
                "counter TP strictly improves; total FP <= baseline; "
                "no new empty-counter or ID cash FP"
            ),
            "promotion": False,
        },
    )
    detector = ObjectDetector(ROOT / DEFAULT_MODEL, 0.25, 768, 0)
    baseline = read(DATA / "resolution-validation.json")["768"]
    frames = []
    for item in baseline["frames"]:
        pixels = cv2.imdecode(np.fromfile(ROOT / item["image"], np.uint8), cv2.IMREAD_COLOR)
        height, width = pixels.shape[:2]
        left, top = np.floor(lower * [width, height]).astype(int)
        right, bottom = np.ceil(upper * [width, height]).astype(int)
        added = []
        for pred in detector.detect(pixels[top:bottom, left:right]):
            box = [v + int(left if i % 2 == 0 else top) for i, v in enumerate(pred["box_xyxy"])]
            center = [(box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height]
            if point_in_polygon(center, polygon):
                added.append({**pred, "box_xyxy": box})
        # Preserve every baseline observation. Suppress only duplicate additions;
        # otherwise an unrelated NMS change could masquerade as a crop benefit.
        merged = list(item["detections"])
        for pred in sorted(added, key=lambda p: p["confidence"], reverse=True):
            if not any(iou(pred["box_xyxy"], q["box_xyxy"]) >= 0.5 for q in merged):
                merged.append(pred)
        frames.append({**item, "detections": merged, "crop_predictions": added})
    result = score_frames(frames)
    old = baseline["metrics"]
    gates = {
        "counter_recall_improves": result["cash_categories"]["cash_at_counter"]["tp"]
        > old["cash_categories"]["cash_at_counter"]["tp"],
        "no_extra_cash_fp": result["classes"]["cash"]["fp"] <= old["classes"]["cash"]["fp"],
        "no_cash_recall_loss": result["classes"]["cash"]["tp"] >= old["classes"]["cash"]["tp"],
        "no_extra_empty_counter_fp": result["false_positives"].get("empty_counter_fp", 0)
        <= old["false_positives"].get("empty_counter_fp", 0),
        "no_extra_id_cash_fp": result["false_positives"].get("id_to_cash_predictions", 0)
        <= old["false_positives"].get("id_to_cash_predictions", 0),
    }
    write(
        DATA / "counter-crop-validation.json",
        {
            "metrics": result,
            "gates": gates,
            "frames": frames,
            "eligible_for_pipeline_trial": all(gates.values()),
            "production_promoted": False,
        },
    )
    print(result, gates, flush=True)


def candidate_id():
    # This is a raw-detection audit, NOT a declaration that id_detector_validated=True.
    detector = ObjectDetector(CANDIDATE, 0.25, 768, 0)
    validation_hashes = {
        i["video_sha256"]
        for i in read(ROOT / "datasets/cash_id_v1/annotations.json")["items"]
        if i["split"] == "val"
    }
    summary = []
    for source in read(MANIFEST):
        if source["video_sha256"] not in validation_hashes:
            continue
        cache = infer(Path(source["video"]), read(cache_path(source)), detector)
        output = DATA / "id-candidate" / source["id"]
        write(output / "detections.json", cache)
        baseline = read(
            ROOT / "data/verdict_separation_v1" / source["id"] / "legacy/track_history.json"
        )
        cfg = EventConfig.model_validate(read(ROOT / source["config"]))
        tracker, associator = LogicalObjectTracker(cfg), EventDetector(cfg)
        frames = []
        for raw, tracked in zip(cache["frames"], baseline["frames"], strict=True):
            objects, _ = suppress_object_duplicates(raw["objects"], cfg.object_duplicate_iou)
            ids = [
                o for o in tracker.update(objects, raw["timestamp_sec"]) if o["label"] == "id_card"
            ]
            frame = FrameObservation.model_validate(
                {"timestamp_sec": raw["timestamp_sec"], "people": tracked["people"], "objects": []}
            )
            associator.active_customer = tracked["current_customer"]
            associator.cashier_id = next(
                (p.track_id for p in frame.people if p.role == "cashier"), None
            )
            for obj in ids:
                obj.update(associator.association(obj, frame))
            frames.append(
                {
                    "timestamp_sec": raw["timestamp_sec"],
                    "frame_index": raw["frame_index"],
                    "ids": ids,
                    "people": tracked["people"],
                }
            )
        write(output / "id-observations.json", frames)
        series = {}
        for frame in frames:
            for obj in frame["ids"]:
                series.setdefault(obj["logical_object_id"], []).append(obj)
        temporal = []
        for identity, nodes in series.items():
            runs = []
            for node in nodes:
                if (
                    not runs
                    or runs[-1][-1]["association"] != node["association"]
                    or node["timestamp_sec"] - runs[-1][-1]["timestamp_sec"]
                    > cfg.max_observation_gap_sec
                ):
                    runs.append([])
                runs[-1].append(node)
            temporal.append(
                {
                    "logical_id": identity,
                    "runs": [
                        {
                            "owner": r[0]["association"],
                            "start": r[0]["timestamp_sec"],
                            "end": r[-1]["timestamp_sec"],
                            "samples": len(r),
                            "stable": len(r) >= cfg.object_min_frames
                            and r[-1]["timestamp_sec"] - r[0]["timestamp_sec"]
                            >= cfg.object_min_sec,
                        }
                        for r in runs
                    ],
                }
            )
        write(output / "temporal-availability.json", temporal)
        summary.append(
            {
                "video": source["id"],
                "frames_with_id": sum(bool(f["ids"]) for f in frames),
                "total_frames": len(frames),
                "model_sha256": detector.model_sha256,
                "semantic_events_created": [],
                "id_coverage_validated": False,
                "temporal_tracks": temporal,
                "reason": "raw class prediction is not transfer/check/table-placement evidence",
            }
        )
    write(DATA / "id-candidate/summary.json", summary)


def review():
    """Evidence selection for human audit only; never feeds inference/event decisions."""
    truth = read(ROOT / "datasets/counter_sequence_v1/ground-truth.json")["videos"]
    for source in read(MANIFEST):
        if source["id"] not in truth:
            continue
        video = cv2.VideoCapture(source["video"])
        panels = []
        actions = list(truth[source["id"]]["actions"])
        if "customer_return_interval" in truth[source["id"]]:
            a, b = truth[source["id"]]["customer_return_interval"]
            actions.append({"id": "customer return", "start_sec": a, "end_sec": b})
        for action in actions:
            for t in np.linspace(action["start_sec"] - 0.4, action["end_sec"] + 0.4, 6):
                video.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, pixels = video.read()
                if not ok:
                    continue
                frame_index = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                # Show raw pixels, not a box from a different sampled frame.
                crop = pixels[260:530, :360].copy()
                cv2.putText(
                    crop,
                    f"{action['id']} frame {frame_index} ({t:.2f}s)",
                    (4, 20),
                    0,
                    0.43,
                    (0, 0, 255),
                    1,
                )
                panels.append(crop)
                imwrite(DATA / source["id"] / "audit-evidence" / f"{frame_index:05d}.jpg", pixels)
        video.release()
        if panels:
            while len(panels) % 3:
                panels.append(np.zeros_like(panels[0]))
            imwrite(
                DATA / source["id"] / "audit-contact-sheet.jpg",
                np.vstack([np.hstack(panels[i : i + 3]) for i in range(0, len(panels), 3)]),
            )


def report():
    audit = read(DATA / "audit-before.json")
    (DATA / "findings.txt").write_text(
        (ROOT / "docs/real-event-audit.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    summary = []
    for source in read(MANIFEST):
        output = DATA / source["id"]
        api, legacy = read(output / "api-result.json"), read(output / "legacy/events.json")
        history = read(output / "legacy/track_history.json")["frames"]
        objects = [o for f in history for o in f["objects"] if o["label"] == "cash"]
        baseline_history = read(
            ROOT / "data/verdict_separation_v1" / source["id"] / "legacy/track_history.json"
        )["frames"]
        baseline_objects = [
            o for f in baseline_history for o in f["objects"] if o["label"] == "cash"
        ]
        write(output / "baseline-cash-observations.json", baseline_objects)
        row = {
            "video": source["id"],
            "verdict": api["verdict"],
            "violations": api["violations"],
            "events": api["diagnostics"]["semantic_events"],
            "cash_operations": api["diagnostics"]["cash_operation"],
            "cash_samples": len(objects),
            "baseline_cash_samples": len(baseline_objects),
            "counter_samples": sum(o["zone"] == "counter" for o in objects),
            "sessions": legacy["rule_result"]["sessions"],
            "insufficient_evidence": api["diagnostics"]["insufficient_evidence_reasons"],
            "timing": read(output / "timing.json"),
        }
        summary.append(row)
        write(output / "semantic-events.json", row["events"])
        write(
            output / "session-regression.json",
            {
                "unchanged": row["sessions"]
                == read(ROOT / "data/verdict_separation_v1" / source["id"] / "legacy/events.json")[
                    "rule_result"
                ]["sessions"]
            },
        )
    changed = [
        p
        for p, digest in read(DATA / "protected-before.json").items()
        if sha256(ROOT / p) != digest
    ]
    tests = ET.parse(DATA / "tests.xml").getroot().find("testsuite").attrib
    validation = read(DATA / "resolution-validation.json")
    six = set(
        read(ROOT / "datasets/cash_id_v2/evaluation-policy.json")["subsets"]["counter_without_hand"]
    )
    six_results = {
        k: score_frames([f for f in v["frames"] if f["id"] in six]) for k, v in validation.items()
    }
    write(DATA / "counter-six.json", six_results)
    crop = read(DATA / "counter-crop-validation.json")
    write(
        DATA / "summary.json",
        {
            "videos": summary,
            "protected_changed": changed,
            "tests": tests,
            "decision": "Both inference variants rejected; no production fix promoted",
            "counter_crop_eligible": crop["eligible_for_pipeline_trial"],
            "real_violation_goal_achieved": False,
        },
    )

    def esc(value):
        return html.escape(str(value))

    document = """<!doctype html><meta charset="utf-8"><title>Sana real-event audit</title>
<style>body{font:16px/1.5 system-ui;margin:30px;max-width:1400px}
td,th{border:1px solid #bbb;padding:8px;vertical-align:top}table{border-collapse:collapse}
pre{white-space:pre-wrap;background:#eee;padding:15px}img{max-width:100%}</style>"""
    document += "<h1>R04 / R05 / R03: real observations, one-layer inference experiment</h1>"
    document += """<h2>Результат: автоматические R04/R03/R05 пока не восстановлены</h2>
<p>Оба варианта inference отклонены. Основные веса, Event Detector, tracking,
калибровка и правила не менялись. Новых подтверждённых нарушений нет.</p>
<p>1280 vs 768, same production cash weights, confidence .25. Person/pose inference reused;
downstream processing rerun. No rule/production changes. Repeated development validation,
not independent evaluation. Previous track-linked diagnostics preserved separately,
never rebound to new IDs. Timing excludes cached person/pose inference and model loading.</p>
<p><a href="findings.txt">Подробные выводы и следующий минимальный шаг</a> |
<a href="tests.xml">Регрессионные тесты</a></p><table><tr>"""
    document += "".join("<th>" + esc(c) + "</th>" for c in audit["columns"]) + "</tr>"
    document += (
        "".join(
            "<tr>" + "".join("<td>" + esc(c) + "</td>" for c in row) + "</tr>"
            for row in audit["rows"]
        )
        + "</table>"
    )
    for row in summary:
        name = row["video"]
        document += (
            f"<h2>{name.upper()}</h2><p>{row['verdict']}; "
            f"violations={len(row['violations'])}; cash samples={row['cash_samples']}; "
            f"counter samples={row['counter_samples']}</p>"
        )
        document += (
            f'<p><a href="{name}/legacy/debug.mp4">Debug video</a> | '
            f'<a href="{name}/api-result.json">API + diagnostics</a> | '
            f'<a href="{name}/semantic-events.json">Real events</a></p>'
        )
        if (DATA / name / "audit-contact-sheet.jpg").exists():
            document += f'<p><a href="{name}/audit-contact-sheet.jpg">Raw audit frames</a></p>'
        document += (
            "<details><summary>Events, timing, diagnostics</summary><pre>"
            + esc(json.dumps(row, indent=2, ensure_ascii=False))
            + "</pre></details>"
        )
    document += (
        "<h2>Training overlap</h2><pre>"
        + esc(json.dumps(read(DATA / "training-overlap.json"), indent=2))
        + "</pre>"
    )
    document += (
        "<h2>Resolution validation (42 frames)</h2><pre>"
        + esc(
            json.dumps(
                {k: v["metrics"] for k, v in read(DATA / "resolution-validation.json").items()},
                indent=2,
            )
        )
        + "</pre>"
    )
    document += (
        "<h2>Counter crop: rejected, no counter recall gain</h2><pre>"
        + esc(json.dumps({"metrics": crop["metrics"], "gates": crop["gates"]}, indent=2))
        + "</pre>"
    )
    document += (
        "<h2>ID candidate</h2><pre>"
        + esc(json.dumps(read(DATA / "id-candidate/summary.json"), indent=2))
        + "</pre>"
    )
    document += '''<h2>Evidence frames</h2><ul>
<li><a href="x1/audit-contact-sheet.jpg">X-1: загрузка, извлечение, возврат клиенту</a></li>
<li><a href="../counter_hand_v1/x1/contact-1.jpg">Pose: контакт / отход / появление денег</a></li>
<li><a href="../table_cash_continuity_v1/x4/evidence/cash-20--cash-22.jpg">
X-4: разрыв cash-20 → cash-22</a></li>
<li><a href="../table_cash_continuity_v1/report.html">Полный прежний R05 continuity audit</a></li>
<li><a href="id-candidate/x3/review.jpg">X-3: ID detection и ошибки ownership</a></li>
<li><a href="id-candidate/x2/review.jpg">X-2: деньги ошибочно определены как ID</a></li></ul>'''
    document += "<p>Protected files changed: " + esc(changed) + "</p>"
    document += "<p>Regression tests: " + esc(tests) + "</p>"
    (DATA / "report.html").write_text(document, encoding="utf-8")
    if changed:
        raise AssertionError(changed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "review",
            "pipeline",
            "validation",
            "counter_crop",
            "candidate_id",
            "report",
        ],
    )
    globals()[parser.parse_args().command]()
