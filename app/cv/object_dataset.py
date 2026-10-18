"""Audit and export a two-class dataset with a sealed, source-disjoint holdout."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from app.cv.cash_dataset import reviewed_items
from app.cv.evaluate_cash import iou

CLASSES = ["cash", "id_card"]
CASH_CATEGORIES = {"cash_in_hand", "cash_transfer", "cash_at_counter", "cash_on_table"}
ID_CATEGORIES = {"id_customer_hand", "id_transfer", "id_cashier_hand", "id_table"}
NEGATIVES = {"empty_counter", "hands_without_cash", "table_without_cash", "id_without_cash"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validated_items(data: dict) -> list[dict]:
    if data.get("classes") != CLASSES:
        raise ValueError("Exactly cash and id_card classes required, in that order")
    if data.get("audit_complete") is not True:
        raise ValueError("Annotation audit must be complete before export/evaluation")
    converted = []
    for item in data["items"]:
        if item.get("review") not in {"reviewed", "reviewed_provisional"}:
            converted.append({**item, "boxes": None})
            continue
        objects = item.get("objects")
        if not isinstance(objects, list):
            raise ValueError("Reviewed frame requires explicit objects (empty for a negative)")
        for index, obj in enumerate(objects):
            label = obj["label"]
            if label not in CLASSES:
                raise ValueError(f"Unknown class: {label}")
            categories = CASH_CATEGORIES if label == "cash" else ID_CATEGORIES
            if obj.get("category") not in categories:
                raise ValueError(f"Missing/invalid diagnostic category: {item['id']}")
            for other in objects[:index]:
                if iou(obj["box_xyxy"], other["box_xyxy"]) > 0.90:
                    raise ValueError(f"Duplicate/conflicting annotations: {item['id']}")
        if set(item.get("categories", [])) - NEGATIVES:
            raise ValueError("Unknown negative category")
        if "empty_counter" in item.get("categories", []):
            roi = item.get("counter_roi")
            if roi is None:
                raise ValueError("Empty counter needs a reviewed ROI")
            # An ROI is evaluation metadata, never a detector input.
            reviewed_items({"class": "cash", "items": [{**item, "boxes": [roi]}]})
            for obj in objects:
                x1, y1, x2, y2 = obj["box_xyxy"]
                if obj["label"] == "cash" and (
                    roi[0] <= (x1 + x2) / 2 <= roi[2] and roi[1] <= (y1 + y2) / 2 <= roi[3]
                ):
                    raise ValueError("Empty counter ROI contains annotated cash")
        converted.append({**item, "boxes": [o["box_xyxy"] for o in objects]})
    selected_ids = {i["id"] for i in reviewed_items({"class": "cash", "items": converted})}
    return [i for i in data["items"] if i["id"] in selected_ids]


def split_digest(items: list[dict], split: str) -> str:
    """Bind pixels, labels, categories, source identity and sampling to the holdout."""
    rows = [
        {
            k: i.get(k)
            for k in (
                "id",
                "video_sha256",
                "image_sha256",
                "timecode_sec",
                "width",
                "height",
                "objects",
                "categories",
                "counter_roi",
            )
        }
        for i in items
        if i["split"] == split
    ]
    payload = json.dumps(sorted(rows, key=lambda i: i["id"]), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_files(items: list[dict]) -> None:
    from PIL import Image

    sources, images = {}, {}
    for item in items:
        video, image = Path(item["video"]), Path(item["image"])
        if video not in sources:
            sources[video] = sha256(video)
        if sources[video] != item["video_sha256"]:
            raise ValueError(f"Source changed: {video}")
        digest = sha256(image)
        if digest != item.get("image_sha256"):
            raise ValueError(f"Image changed/unsealed: {image}")
        if images.setdefault(digest, item["split"]) != item["split"]:
            raise ValueError("Duplicate image leakage across splits")
        with Image.open(image) as frame:
            if frame.size != (item["width"], item["height"]):
                raise ValueError(f"Dimensions changed: {image}")


def verify_holdout(data: dict, items: list[dict]) -> None:
    if split_digest(items, "val") != data.get("validation_sha256"):
        raise ValueError("Fixed validation changed or has not been sealed")
    holdout = [i for i in items if i["split"] == "val"]
    if not holdout or not any(i["split"] == "train" for i in items):
        raise ValueError("Train and validation must both be nonempty")
    coverage = {o["category"] for i in holdout for o in i["objects"]}
    coverage |= {c for i in holdout for c in i.get("categories", [])}
    required = {
        "cash_in_hand",
        "cash_transfer",
        "cash_at_counter",
        "empty_counter",
        "hands_without_cash",
        "table_without_cash",
    }
    if required - coverage or not any(
        o["label"] == "id_card" for i in holdout for o in i["objects"]
    ):
        raise ValueError("Validation coverage is incomplete")
    prior = set(data.get("baseline_training_video_sha256", []))
    if not prior:
        raise ValueError("Baseline training-source inventory required")
    if prior & {i["video_sha256"] for i in holdout}:
        raise ValueError("Validation source was seen in baseline training")


def summary(items: list[dict]) -> dict:
    return {
        split: {
            "images": len(rows := [i for i in items if i["split"] == split]),
            "source_videos": len({i["video_sha256"] for i in rows}),
            "classes": dict(Counter(o["label"] for i in rows for o in i["objects"])),
            "object_categories": dict(Counter(o["category"] for i in rows for o in i["objects"])),
            "negative_categories": dict(Counter(c for i in rows for c in i.get("categories", []))),
        }
        for split in ("train", "val", "test")
    }


def export_dataset(annotations: Path, output: Path) -> dict:
    data = json.loads(annotations.read_text(encoding="utf-8"))
    items = validated_items(data)
    verify_holdout(data, items)
    verify_files(items)
    output.mkdir(parents=True, exist_ok=False)
    for item in items:
        images, labels = output / "images" / item["split"], output / "labels" / item["split"]
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["image"], images / f"{item['id']}.jpg")
        w, h = item["width"], item["height"]
        lines = []
        for obj in item["objects"]:
            x1, y1, x2, y2 = obj["box_xyxy"]
            lines.append(
                f"{CLASSES.index(obj['label'])} {(x1 + x2) / (2 * w):.8f} "
                f"{(y1 + y2) / (2 * h):.8f} {(x2 - x1) / w:.8f} {(y2 - y1) / h:.8f}"
            )
        (labels / f"{item['id']}.txt").write_text("\n".join(lines), encoding="utf-8")
    config = {
        "path": output.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(CLASSES)),
    }
    (output / "objects.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {
        "annotation_sha256": sha256(annotations),
        "validation_sha256": data["validation_sha256"],
        "summary": summary(items),
        "annotation_provenance": data["annotation_provenance"],
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copyfile(annotations, output / "manifest.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.annotations, args.output), indent=2))


if __name__ == "__main__":
    main()
