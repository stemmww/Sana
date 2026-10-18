"""Independent counter occupancy experiment; never imported by the video pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/counter_occupancy_v1"
META = ROOT / "datasets/counter_occupancy_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def imread(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def imwrite(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(path)


def crop_zone(image, zone):
    h, w = image.shape[:2]
    points = np.rint(np.array(zone) * [w, h]).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    x, y, cw, ch = cv2.boundingRect(points)
    mask = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(mask, [points - [x, y]], 255)
    crop = image[y : y + ch, x : x + cw].copy()
    return crop, mask, [x, y, x + cw, y + ch]


def sheet(items, path, predictions=None):
    cols, tw, th = 4, 285, 355
    canvas = Image.new("RGB", (cols * tw, ((len(items) + cols - 1) // cols) * th), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for n, item in enumerate(items):
        x, y = (n % cols) * tw, (n // cols) * th
        crop = Image.open(ROOT / item["crop"]).convert("RGB")
        crop.thumbnail((255, 300))
        # Nearest-neighbor enlargement exposes the actual available pixels.
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
        crop.thumbnail((255, 300))
        canvas.paste(crop, (x + 8, y + 46))
        title = f"{item['id']} {item.get('label', 'UNREVIEWED')}"
        draw.text((x + 8, y + 5), title, fill="black")
        if predictions and item["id"] in predictions:
            pred = predictions[item["id"]]
            draw.text((x + 8, y + 23), str(pred), fill="#a00000")
    canvas.save(path)


def freeze():
    paths = set(ROOT.glob("app/**/*.py")) | set(ROOT.glob("config/**/*"))
    paths |= set(ROOT.glob("datasets/cash_id_v*/annotations.json"))
    paths |= {p for p in ROOT.rglob("*.pt") if ".venv" not in p.parts}
    paths = {p for p in paths if p.is_file()}
    write_json(
        META / "protected-sha256.json", {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    )


def prepare():
    if (META / "annotations.json").exists():
        raise FileExistsError("Reviewed annotations already exist; refusing to overwrite them.")
    DATA.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    freeze()
    zone = read_json(ROOT / "config/event_camera.json")["zones"]["counter"]
    sources = read_json(ROOT / "data/cash_id_v2/scouted-training-sources.json")
    # Sampling plan is dataset metadata, never an inference feature or video-specific rule.
    plan = {
        1: [0, 5, 6, 9.5, 12, 14, 14.5, 15, 16, 18, 19, 20, 20.5, 22, 24],
        4: [0, 13, 16, 19, 21, 24, 25, 26, 27, 29.5, 31, 32, 33, 40, 50, 60, 65, 70],
        5: [0, 8, 12, 19, 21, 24, 26, 27, 28, 30, 31, 32, 34, 50, 69, 70],
    }
    items, excluded_sources = [], []
    for source in sources:
        number = source["video_number"]
        if number not in plan:
            excluded_sources.append(
                {
                    **source,
                    "reason": (
                        "Camera view does not place the whole counter "
                        "in the existing configured zone."
                    ),
                }
            )
            continue
        assert sha(source["video"]) == source["video_sha256"]
        cap = cv2.VideoCapture(source["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        for sec in plan[number]:
            idx = round(sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise ValueError((number, sec))
            ident = f"c{number:02d}_{round(sec * 100):05d}"
            frame_path = DATA / "frames" / f"{ident}.png"
            crop_path = DATA / "crops" / f"{ident}.png"
            crop, mask, bbox = crop_zone(frame, zone)
            imwrite(frame_path, frame)
            imwrite(crop_path, cv2.bitwise_and(crop, crop, mask=mask))
            imwrite(DATA / "zone-mask.png", mask)
            items.append(
                {
                    "id": ident,
                    "split": "dev" if number == 5 else "train",
                    "video": source["video"],
                    "video_sha256": source["video_sha256"],
                    "timecode_sec": idx / fps,
                    "frame_index": idx,
                    "image": str(frame_path.relative_to(ROOT)),
                    "image_sha256": sha(frame_path),
                    "crop": str(crop_path.relative_to(ROOT)),
                    "crop_sha256": sha(crop_path),
                    "zone_bbox_xyxy": bbox,
                    "label": "UNREVIEWED",
                    "note": "",
                }
            )
        cap.release()
        subset = [i for i in items if i["video_sha256"] == source["video_sha256"]]
        sheet(subset, DATA / f"train-review-{number}.jpg")
    manifest = {
        "schema_version": 1,
        "zone_source": "config/event_camera.json:zones.counter",
        "zone": zone,
        "labels": ["EMPTY", "CASH_PRESENT", "UNCERTAIN"],
        "annotation_provenance": (
            "Assistant visual review; provisional, pending human confirmation."
        ),
        "split_policy": (
            "Source-disjoint: X-1/X-4 train, X-5 dev; "
            "original 42 X-3/X-2 frames final evaluation only."
        ),
        "excluded_sources": excluded_sources,
        "items": items,
    }
    write_json(META / "annotations.json", manifest)
    print(f"Prepared {len(items)} training-source crops. No validation crops used.")


def checked_manifest():
    manifest = read_json(META / "annotations.json")
    assert manifest["zone"] == read_json(ROOT / "config/event_camera.json")["zones"]["counter"]
    original = read_json(ROOT / "datasets/cash_id_v2/annotations.json")
    held_sources = {i["video_sha256"] for i in original["items"] if i["split"] == "val"}
    train = {i["video_sha256"] for i in manifest["items"] if i["split"] == "train"}
    dev = {i["video_sha256"] for i in manifest["items"] if i["split"] == "dev"}
    assert not train & dev
    assert not (train | dev) & held_sources
    for item in manifest["items"]:
        assert item["label"] in manifest["labels"]
        assert item["crop_sha256"] == sha(ROOT / item["crop"])
        assert item["image_sha256"] == sha(ROOT / item["image"])
    return manifest


def selected(items, split):
    return [
        i
        for i in items
        if i["split"] == split and i["label"] != "UNCERTAIN" and i.get("selected", True)
    ]


def binary_metrics(items, scores, threshold):
    if any(i["label"] not in {"EMPTY", "CASH_PRESENT"} for i in items):
        raise ValueError("Ambiguous/unreviewed labels must not enter binary metrics.")
    if len(items) != len(scores):
        raise ValueError("One score is required for every frame.")
    y = np.array([i["label"] == "CASH_PRESENT" for i in items])
    pred = np.asarray(scores) >= threshold
    tp, fn = int(np.sum(y & pred)), int(np.sum(y & ~pred))
    fp, tn = int(np.sum(~y & pred)), int(np.sum(~y & ~pred))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "errors": [i["id"] for i, a, b in zip(items, y, pred, strict=True) if a != b],
    }


def select_threshold(items, scores):
    # Dev only. Equal cost per class; tie favors fewer false cash and larger margin.
    unique = np.unique(scores)
    thresholds = [
        float(unique[0] - 1e-6),
        *map(float, (unique[:-1] + unique[1:]) / 2),
        float(unique[-1] + 1e-6),
    ]
    results = [(t, binary_metrics(items, scores, t)) for t in thresholds]
    return max(
        results,
        key=lambda x: (x[1]["balanced_accuracy"], -x[1]["fp"], min(abs(np.array(scores) - x[0]))),
    )


def cv_features(image, reference, mask):
    """OpenCV only: empty-reference difference, color, local texture. No video/time input."""
    image = cv2.GaussianBlur(image, (3, 3), 0)
    reference = cv2.GaussianBlur(reference, (3, 3), 0)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Compensate whole-zone exposure/color drift using a robust median; cash is small.
    shift = np.median((lab - ref)[mask], axis=0)
    delta = lab - ref - shift
    diff = np.sqrt(np.mean(delta**2, axis=2))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    features = []
    h, w = mask.shape
    for row in range(4):
        for col in range(4):
            cell = np.zeros_like(mask)
            cell[row * h // 4 : (row + 1) * h // 4, col * w // 4 : (col + 1) * w // 4] = True
            cell &= mask
            if not np.any(cell):
                features.extend([0] * 6)
                continue
            features.extend(
                [
                    np.mean(diff[cell]),
                    np.percentile(diff[cell], 90),
                    np.mean(hsv[:, :, 1][cell]),
                    np.std(gray[cell]),
                    np.mean(edges[cell]),
                    np.mean(delta[:, :, 0][cell]),
                ]
            )
    scores = [
        float(np.mean(np.abs(image.astype(float) - reference.astype(float))[mask])),
        float(np.percentile(diff[mask], 98)),
    ]
    return np.array(features, np.float32), scores


def fit_opencv():
    manifest = checked_manifest()
    train = selected(manifest["items"], "train")
    dev = selected(manifest["items"], "dev")
    mask = imread(DATA / "zone-mask.png")[:, :, 0] > 0
    empty = [imread(ROOT / i["crop"]) for i in train if i["label"] == "EMPTY"]
    reference = np.median(empty, axis=0).astype(np.uint8)
    imwrite(DATA / "empty-reference.png", reference)
    all_items = train + dev
    results = [cv_features(imread(ROOT / i["crop"]), reference, mask) for i in all_items]
    features = np.array([r[0] for r in results])
    scores = np.array([r[1] for r in results])
    n = len(train)
    models = []
    for idx, name in enumerate(["absolute_difference", "compensated_difference_p98"]):
        threshold, metrics = select_threshold(dev, scores[n:, idx])
        models.append(
            {
                "name": name,
                "type": "difference",
                "score_index": idx,
                "threshold": threshold,
                "train": binary_metrics(train, scores[:n, idx], threshold),
                "dev": metrics,
            }
        )
    mean, scale = features[:n].mean(axis=0), features[:n].std(axis=0).clip(1)
    x = (features - mean) / scale
    y = np.array([1 if i["label"] == "CASH_PRESENT" else -1 for i in train], np.float32)
    # Class-balanced ridge head over fixed OpenCV features; no neural network or new dependencies.
    weights = np.array([n / (2 * np.sum(y == label)) for label in y])
    for alpha in [1.0, 10.0, 100.0]:
        xx = np.column_stack([x, np.ones(len(x))])
        penalty = np.eye(xx.shape[1]) * alpha
        penalty[-1, -1] = 0
        coef = np.linalg.solve((xx[:n].T * weights) @ xx[:n] + penalty, (xx[:n].T * weights) @ y)
        decision = xx @ coef
        threshold, metrics = select_threshold(dev, decision[n:])
        models.append(
            {
                "name": f"color_texture_ridge_{alpha:g}",
                "type": "ridge",
                "alpha": alpha,
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "coef": coef.tolist(),
                "threshold": threshold,
                "train": binary_metrics(train, decision[:n], threshold),
                "dev": metrics,
            }
        )
    best = max(
        models,
        key=lambda m: (
            m["dev"]["balanced_accuracy"],
            -m["dev"]["fp"],
            m["train"]["balanced_accuracy"],
            m["type"] == "difference",
            m.get("alpha", 0),
        ),
    )
    artifact = {
        "reference_sha256": sha(DATA / "empty-reference.png"),
        "annotation_sha256": sha(META / "annotations.json"),
        "zone": manifest["zone"],
        "train_ids": [i["id"] for i in train],
        "dev_ids": [i["id"] for i in dev],
        "selection": (
            "Dev balanced accuracy, then false positives, train balanced accuracy, "
            "simplest model, stronger regularization; no fixed validation accessed."
        ),
        "models": models,
        "selected_model": best["name"],
        "fallback_trigger": "Selected OpenCV dev recall < 1.0 or dev specificity < 1.0.",
    }
    write_json(DATA / "opencv-models.json", artifact)
    write_json(
        META / "opencv-dev-results.json",
        {
            "selected_model": best["name"],
            "models": [
                {k: v for k, v in m.items() if k not in {"mean", "scale", "coef"}} for m in models
            ],
        },
    )
    print(json.dumps(read_json(META / "opencv-dev-results.json"), indent=2))


def prepare_validation():
    # Call only after fitting/freezing the candidate on training-source videos.
    artifact = read_json(DATA / "opencv-models.json")
    assert artifact["annotation_sha256"] == sha(META / "annotations.json")
    source = read_json(ROOT / "datasets/cash_id_v2/annotations.json")
    six = {"v03_01850", "v03_01900", "v03_01950", "v02_01550", "v02_01600", "v02_01650"}
    items = []
    for original in source["items"]:
        if original["split"] != "val" or original.get("review") not in {
            "reviewed",
            "reviewed_provisional",
        }:
            continue
        assert sha(ROOT / original["image"]) == original["image_sha256"]
        crop, mask, bbox = crop_zone(imread(ROOT / original["image"]), artifact["zone"])
        crop_path = DATA / "validation-crops" / f"{original['id']}.png"
        imwrite(crop_path, cv2.bitwise_and(crop, crop, mask=mask))
        item = {
            k: original[k]
            for k in ["id", "video", "video_sha256", "timecode_sec", "image", "image_sha256"]
        }
        subset = (
            "six_counter_without_hand"
            if item["id"] in six
            else "empty_counter_28"
            if "empty_counter" in original.get("categories", [])
            else "other_eight"
        )
        item.update(
            {
                "split": "validation",
                "subset": subset,
                "label": "UNREVIEWED",
                "note": "",
                "crop": str(crop_path.relative_to(ROOT)),
                "crop_sha256": sha(crop_path),
                "zone_bbox_xyxy": bbox,
            }
        )
        items.append(item)
    assert len(items) == 42
    assert sum(i["subset"] == "empty_counter_28" for i in items) == 28
    write_json(
        META / "validation.json",
        {
            "source_sha256": sha(ROOT / "datasets/cash_id_v2/annotations.json"),
            "frozen_opencv_models_sha256": sha(DATA / "opencv-models.json"),
            "items": items,
        },
    )
    for n in range(0, len(items), 12):
        sheet(items[n : n + 12], DATA / f"validation-review-{n // 12}.jpg")
    print("42 original validation frames cropped; no new video frames sampled.")


def model_scores(model, features, differences=None):
    if model["type"] == "difference":
        return np.asarray(differences)[:, model["score_index"]]
    x = (np.asarray(features) - np.array(model["mean"])) / np.array(model["scale"])
    return np.column_stack([x, np.ones(len(x))]) @ np.array(model["coef"])


def evaluate_scores(items, model, scores):
    rows = []
    for item, score in zip(items, scores, strict=True):
        pred = "CASH_PRESENT" if score >= model["threshold"] else "EMPTY"
        result = (
            "UNSCORED"
            if item["label"] == "UNCERTAIN"
            else "OK"
            if item["label"] == pred
            else "FN"
            if item["label"] == "CASH_PRESENT"
            else "FP"
        )
        rows.append(
            {
                "id": item["id"],
                "timecode_sec": item["timecode_sec"],
                "subset": item["subset"],
                "label": item["label"],
                "prediction": pred,
                "score": float(score),
                "result": result,
            }
        )
    groups = {"all_clear": [n for n, i in enumerate(items) if i["label"] != "UNCERTAIN"]}
    for subset in ["six_counter_without_hand", "empty_counter_28"]:
        groups[subset] = [n for n, i in enumerate(items) if i["subset"] == subset]
    metrics = {
        name: binary_metrics([items[n] for n in idx], np.array(scores)[idx], model["threshold"])
        for name, idx in groups.items()
    }
    return {
        "model": model["name"],
        "threshold": model["threshold"],
        "metrics": metrics,
        "rows": rows,
    }


def evaluate_opencv():
    checked_manifest()
    validation = read_json(META / "validation.json")
    assert validation["frozen_opencv_models_sha256"] == sha(DATA / "opencv-models.json")
    items = validation["items"]
    reference = imread(DATA / "empty-reference.png")
    mask = imread(DATA / "zone-mask.png")[:, :, 0] > 0
    artifact = read_json(DATA / "opencv-models.json")
    assert artifact["reference_sha256"] == sha(DATA / "empty-reference.png")
    results = [cv_features(imread(ROOT / i["crop"]), reference, mask) for i in items]
    evaluations = []
    for model in artifact["models"]:
        scores = model_scores(model, [r[0] for r in results], [r[1] for r in results])
        evaluations.append(evaluate_scores(items, model, scores))
    write_json(
        META / "opencv-validation-results.json",
        {"selected_model": artifact["selected_model"], "evaluations": evaluations},
    )
    selected_result = next(e for e in evaluations if e["model"] == artifact["selected_model"])
    print(
        json.dumps(
            {"selected_model": artifact["selected_model"], "metrics": selected_result["metrics"]},
            indent=2,
        )
    )


def audit():
    manifest = checked_manifest()
    protected = read_json(META / "protected-sha256.json")
    changed = [path for path, digest in protected.items() if sha(ROOT / path) != digest]
    assert not changed, changed
    val = read_json(META / "validation.json")
    original = read_json(ROOT / "datasets/cash_id_v2/annotations.json")
    expected = {
        i["id"]: i
        for i in original["items"]
        if i["split"] == "val" and i.get("review") in {"reviewed", "reviewed_provisional"}
    }
    assert set(expected) == {i["id"] for i in val["items"]}
    for i in val["items"]:
        assert i["image_sha256"] == sha(ROOT / i["image"]) == expected[i["id"]]["image_sha256"]
        assert i["crop_sha256"] == sha(ROOT / i["crop"])
        assert i["label"] in manifest["labels"]
        assert i["video_sha256"] == expected[i["id"]]["video_sha256"]
    from collections import Counter

    counts = Counter(
        f"{i['split']}/{i['label']}/selected={i['selected']}" for i in manifest["items"]
    )
    report = {
        "protected_files_verified": len(protected),
        "protected_changed": changed,
        "fixed_validation_original_pixels_verified": len(expected),
        "source_leakage": False,
        "counts": dict(counts),
        "validation_labels": dict(Counter(i["label"] for i in val["items"])),
    }
    write_json(META / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "fit-opencv", "prepare-validation", "evaluate-opencv", "audit"],
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "fit-opencv":
        fit_opencv()
    elif args.command == "prepare-validation":
        prepare_validation()
    elif args.command == "evaluate-opencv":
        evaluate_opencv()
    elif args.command == "audit":
        audit()
