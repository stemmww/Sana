"""Compare object detectors on a sealed holdout, including counter and ID confusions."""

import argparse
import json
from collections import Counter
from pathlib import Path

from app.cv.evaluate_cash import iou
from app.cv.object_dataset import (
    CASH_CATEGORIES,
    CLASSES,
    sha256,
    validated_items,
    verify_files,
    verify_holdout,
)
from app.cv.object_detector import ObjectDetector


def match_indices(predictions, targets, threshold=0.5):
    """Greedy, confidence-ordered, one-to-one, class-aware matching."""
    remaining = set(range(len(targets)))
    matches = {}
    for index in sorted(
        range(len(predictions)), key=lambda i: predictions[i]["confidence"], reverse=True
    ):
        pred = predictions[index]
        candidates = [i for i in remaining if targets[i]["label"] == pred["label"]]
        best = max(
            candidates, key=lambda i: iou(pred["box_xyxy"], targets[i]["box_xyxy"]), default=None
        )
        if best is not None and iou(pred["box_xyxy"], targets[best]["box_xyxy"]) >= threshold:
            matches[index] = best
            remaining.remove(best)
    return matches


def metrics(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def target_coverage(pred_box, target_box):
    """Fraction of a small target covered, independent of prediction-box size."""
    width = max(0, min(pred_box[2], target_box[2]) - max(pred_box[0], target_box[0]))
    height = max(0, min(pred_box[3], target_box[3]) - max(pred_box[1], target_box[1]))
    area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    return width * height / area if area > 0 else 0.0


def score_frames(frames, threshold=0.5):
    totals = {c: Counter() for c in CLASSES}
    category_counts = {c: Counter() for c in sorted(CASH_CATEGORIES)}
    negatives = Counter()
    for frame in frames:
        targets, predictions = frame["objects"], frame["detections"]
        matches = match_indices(predictions, targets, threshold)
        matched_targets = set(matches.values())
        for label in CLASSES:
            tp = sum(targets[t]["label"] == label for t in matched_targets)
            fp = sum(p["label"] == label and i not in matches for i, p in enumerate(predictions))
            fn = sum(
                t["label"] == label and i not in matched_targets for i, t in enumerate(targets)
            )
            totals[label].update(tp=tp, fp=fp, fn=fn)
        for category in category_counts:
            relevant = {i for i, t in enumerate(targets) if t["category"] == category}
            if relevant:
                # Target recall is category-specific; precision includes every unmatched
                # cash prediction in these frames. Categories may overlap across frames.
                tp = len(relevant & matched_targets)
                fp = sum(
                    p["label"] == "cash" and i not in matches for i, p in enumerate(predictions)
                )
                category_counts[category].update(tp=tp, fp=fp, fn=len(relevant) - tp, frames=1)
        cash_fp = [
            p for i, p in enumerate(predictions) if p["label"] == "cash" and i not in matches
        ]
        ids = [t for t in targets if t["label"] == "id_card"]
        strict_confusion = sum(
            any(iou(p["box_xyxy"], t["box_xyxy"]) >= threshold for t in ids) for p in cash_fp
        )
        # Tiny ID targets can be fully covered by a loose cash box with IoU < .5.
        # Keep the strict metric, and separately measure this classification failure.
        confusion = sum(
            any(target_coverage(p["box_xyxy"], t["box_xyxy"]) >= 0.5 for t in ids) for p in cash_fp
        )
        negatives.update(
            id_to_cash_predictions=confusion,
            id_to_cash_iou50_predictions=strict_confusion,
            id_to_cash_frames=int(confusion > 0),
            id_frames=int(bool(ids)),
            id_objects=len(ids),
        )
        if "empty_counter" in frame.get("categories", []):
            roi = frame["counter_roi"]
            inside = 0
            for pred in cash_fp:
                x1, y1, x2, y2 = pred["box_xyxy"]
                inside += int(
                    roi[0] <= (x1 + x2) / 2 <= roi[2] and roi[1] <= (y1 + y2) / 2 <= roi[3]
                )
            negatives.update(
                empty_counter_frames=1,
                empty_counter_fp=inside,
                empty_counter_fp_frames=int(inside > 0),
            )
        for tag in ("hands_without_cash", "table_without_cash", "id_without_cash"):
            if tag in frame.get("categories", []):
                negatives.update({tag + "_frames": 1, tag + "_cash_fp": len(cash_fp)})
    return {
        "classes": {c: metrics(v["tp"], v["fp"], v["fn"]) for c, v in totals.items()},
        "cash_categories": {
            c: {**metrics(v["tp"], v["fp"], v["fn"]), "frames": v["frames"]}
            for c, v in category_counts.items()
        },
        "false_positives": dict(negatives),
    }


def evaluate(annotations: Path, weights: Path, output: Path, confidence=0.25, imgsz=768):
    import cv2

    data = json.loads(annotations.read_text(encoding="utf-8"))
    items = validated_items(data)
    verify_holdout(data, items)
    items = [i for i in items if i["split"] == "val"]
    verify_files(items)
    detector = ObjectDetector(weights, confidence, imgsz)
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    for item in items:
        frame = cv2.imread(item["image"])
        if frame is None:
            raise ValueError(f"Cannot read {item['image']}")
        predictions = detector.detect(frame)
        frames.append({**item, "detections": predictions})
        for obj in item["objects"]:
            x1, y1, x2, y2 = map(round, obj["box_xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
        for pred in predictions:
            x1, y1, x2, y2 = map(round, pred["box_xyxy"])
            color = (0, 220, 0) if pred["label"] == "cash" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                frame,
                f"{pred['label']} {pred['confidence']:.2f}",
                (x1, max(10, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
            )
        if not cv2.imwrite(str(output / f"{item['id']}.jpg"), frame):
            raise OSError("Cannot save evaluation image")
    report = {
        "metrics_version": 2,
        "model": str(weights.resolve()),
        "model_sha256": detector.model_sha256,
        "supported_classes": sorted(detector.supported),
        "unsupported_classes": sorted(set(CLASSES) - detector.supported),
        "annotation_sha256": sha256(annotations),
        "validation_sha256": data["validation_sha256"],
        "annotation_provenance": data["annotation_provenance"],
        "images": len(frames),
        "confidence": confidence,
        "iou": 0.5,
        "imgsz": imgsz,
        "definitions": {
            "empty_counter_fp": "Unmatched cash center inside reviewed empty-counter ROI",
            "id_to_cash": "Unmatched cash prediction covering >=50% of an annotated ID",
            "id_to_cash_iou50": "Unmatched cash with IoU >=0.5 to ID (strict localization)",
            "category_precision": "Category TP / (category TP + cash FP on category frames)",
            "category_recall": "Matched targets / all targets of the diagnostic category",
            "preview": "magenta: annotation; green: predicted cash; cyan: predicted id_card",
        },
        **score_frames(frames),
        "frames": frames,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "frames"}


def compare_reports(old: dict, new: dict) -> dict:
    for key in ("validation_sha256", "metrics_version", "confidence", "iou", "imgsz"):
        if old[key] != new[key]:
            raise ValueError(f"Cannot compare different {key}")
    op, np = old["false_positives"], new["false_positives"]
    oc, nc = old["classes"]["cash"], new["classes"]["cash"]
    old_counter = old["cash_categories"]["cash_at_counter"]["recall"]
    new_counter = new["cash_categories"]["cash_at_counter"]["recall"]
    if old_counter is None or new_counter is None:
        raise ValueError("Counter positives are required for comparison")
    # Explicit local acceptance criteria fixed before candidate evaluation.
    gates = {
        "counter_recall_improved": new_counter > old_counter,
        "cash_fp_increase_at_most_one": nc["fp"] <= oc["fp"] + 1,
        "cash_precision_drop_at_most_5pp": nc["precision"] is not None
        and nc["precision"] >= (oc["precision"] or 0) - 0.05,
        "cash_recall_not_lower": nc["recall"] >= oc["recall"],
        "id_to_cash_at_least_halved_or_zero": np.get("id_to_cash_predictions", 0)
        <= op.get("id_to_cash_predictions", 0) / 2,
        "no_empty_counter_fp": np.get("empty_counter_fp", 0) == 0,
        "id_card_detected": new["classes"]["id_card"]["tp"] > 0,
    }
    return {
        "validation_sha256": old["validation_sha256"],
        "gates": gates,
        "candidate_accepted_for_explicit_pipeline_trial": all(gates.values()),
        "production_weights_replaced": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--compare-to", type=Path, help="Baseline metrics.json on the same holdout")
    args = parser.parse_args()
    report = evaluate(args.annotations, args.model, args.output, args.confidence, args.imgsz)
    if args.compare_to:
        baseline = json.loads(args.compare_to.read_text(encoding="utf-8"))
        comparison = compare_reports(baseline, report)
        comparison.update(
            old_model_sha256=baseline["model_sha256"], new_model_sha256=report["model_sha256"]
        )
        (args.output / "comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        report["comparison"] = comparison
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
