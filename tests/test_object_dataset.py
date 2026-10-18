import copy

import pytest

from app.cv.evaluate_objects import compare_reports, score_frames
from app.cv.object_dataset import CLASSES, split_digest, validated_items


def item():
    return {
        "id": "frame_1",
        "split": "train",
        "video_sha256": "a" * 64,
        "image_sha256": "b" * 64,
        "width": 100,
        "height": 100,
        "review": "reviewed_provisional",
        "categories": [],
        "objects": [{"label": "id_card", "box_xyxy": [10, 10, 20, 20], "category": "id_table"}],
    }


def dataset(rows):
    return {"classes": CLASSES, "audit_complete": True, "items": rows}


def test_audit_gate_and_exactly_two_classes():
    data = dataset([item()])
    data["audit_complete"] = False
    with pytest.raises(ValueError, match="audit"):
        validated_items(data)
    data["audit_complete"] = True
    data["classes"] = ["cash", "id_card", "cash_at_counter"]
    with pytest.raises(ValueError, match="Exactly"):
        validated_items(data)


def test_pending_objects_never_become_negative_and_source_leakage_rejected():
    first = item()
    first.update(review="pending", objects=None)
    assert validated_items(dataset([first])) == []
    second = item()
    second.update(id="frame_2", split="val")
    with pytest.raises(ValueError, match="leakage"):
        validated_items(dataset([first, second]))


def test_duplicate_or_conflicting_label_rejected():
    row = item()
    row["objects"].append(
        {"label": "cash", "box_xyxy": [10, 10, 20, 20], "category": "cash_in_hand"}
    )
    with pytest.raises(ValueError, match="Duplicate/conflicting"):
        validated_items(dataset([row]))


def test_empty_counter_needs_roi_and_cannot_contain_cash_target():
    row = item()
    row["categories"] = ["empty_counter"]
    with pytest.raises(ValueError, match="ROI"):
        validated_items(dataset([row]))
    row["counter_roi"] = [0, 0, 30, 30]
    assert validated_items(dataset([row]))
    row["objects"][0].update(label="cash", category="cash_at_counter")
    with pytest.raises(ValueError, match="contains annotated cash"):
        validated_items(dataset([row]))


def test_validation_seal_changes_for_pixels_labels_categories_and_roi():
    row = item()
    row["split"] = "val"
    digest = split_digest([row], "val")
    for field, replacement in [
        ("image_sha256", "c" * 64),
        ("objects", []),
        ("categories", ["empty_counter"]),
        ("counter_roi", [1, 1, 5, 5]),
    ]:
        changed = copy.deepcopy(row)
        changed[field] = replacement
        assert split_digest([changed], "val") != digest


def pred(label, box, confidence=0.9):
    return dict(label=label, box_xyxy=box, confidence=confidence)


def test_wrong_class_is_fp_and_fn_and_empty_counter_is_counted():
    row = item()
    row.update(
        categories=["empty_counter"],
        counter_roi=[50, 50, 90, 90],
        detections=[pred("cash", [10, 10, 20, 20]), pred("cash", [60, 60, 70, 70])],
    )
    result = score_frames([row])
    assert result["classes"]["cash"]["fp"] == 2
    assert result["classes"]["id_card"]["fn"] == 1
    assert result["classes"]["id_card"]["precision"] is None
    assert result["classes"]["id_card"]["recall"] == 0
    assert result["false_positives"]["id_to_cash_predictions"] == 1
    assert result["false_positives"]["empty_counter_fp"] == 1


def test_duplicates_and_diagnostic_recall_are_not_double_counted():
    row = item()
    row["objects"] = [
        dict(label="cash", box_xyxy=[10, 10, 20, 20], category="cash_at_counter"),
        dict(label="cash", box_xyxy=[60, 60, 70, 70], category="cash_in_hand"),
    ]
    row["detections"] = [pred("cash", [10, 10, 20, 20]), pred("cash", [10, 10, 20, 20], 0.8)]
    result = score_frames([row])
    assert result["classes"]["cash"]["tp"] == 1
    assert result["classes"]["cash"]["fp"] == 1
    assert result["classes"]["cash"]["fn"] == 1
    assert result["cash_categories"]["cash_at_counter"]["recall"] == 1
    assert result["cash_categories"]["cash_in_hand"]["recall"] == 0


def test_comparison_rejects_different_holdout():
    with pytest.raises(ValueError, match="validation_sha256"):
        compare_reports({"validation_sha256": "a"}, {"validation_sha256": "b"})


def test_recall_gain_does_not_override_empty_counter_failure():
    report = {
        "metrics_version": 2,
        "validation_sha256": "a",
        "confidence": 0.25,
        "iou": 0.5,
        "imgsz": 768,
        "classes": {"cash": {"fp": 0, "precision": 1.0, "recall": 0.5}, "id_card": {"tp": 1}},
        "false_positives": {"empty_counter_fp": 0, "id_to_cash_predictions": 0},
        "cash_categories": {"cash_at_counter": {"recall": 0.2}},
    }
    new = copy.deepcopy(report)
    new["cash_categories"]["cash_at_counter"]["recall"] = 0.9
    new["false_positives"]["empty_counter_fp"] = 1
    assert not compare_reports(report, new)["candidate_accepted_for_explicit_pipeline_trial"]


def test_loose_cash_box_covering_small_id_is_a_classification_confusion():
    row = item()
    row["detections"] = [pred("cash", [5, 5, 25, 25])]
    result = score_frames([row])
    assert result["false_positives"]["id_to_cash_predictions"] == 1
    assert result["false_positives"]["id_to_cash_iou50_predictions"] == 0
