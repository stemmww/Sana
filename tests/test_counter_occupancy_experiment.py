"""Guard evaluation denominators and the user's isolated holdout boundary."""

import numpy as np
import pytest

from experiments.counter_occupancy import (
    META,
    binary_metrics,
    checked_manifest,
    evaluate_scores,
    read_json,
    selected,
)


def test_ambiguous_is_not_a_negative_or_a_training_example():
    items = [
        {"id": "positive", "split": "train", "label": "CASH_PRESENT"},
        {"id": "occluded", "split": "train", "label": "UNCERTAIN"},
    ]
    assert [i["id"] for i in selected(items, "train")] == ["positive"]
    with pytest.raises(ValueError, match="Ambiguous"):
        binary_metrics(items, [1, 1], 0.5)


def test_all_frames_retained_but_uncertain_not_in_binary_denominator():
    items = [
        {
            "id": "p",
            "timecode_sec": 1,
            "label": "CASH_PRESENT",
            "subset": "six_counter_without_hand",
        },
        {"id": "n", "timecode_sec": 2, "label": "EMPTY", "subset": "empty_counter_28"},
        {"id": "u", "timecode_sec": 3, "label": "UNCERTAIN", "subset": "other_eight"},
    ]
    result = evaluate_scores(items, {"name": "fixture", "threshold": 0.5}, np.array([0, 1, 1]))
    assert [r["result"] for r in result["rows"]] == ["FN", "FP", "UNSCORED"]
    m = result["metrics"]["all_clear"]
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (0, 1, 1, 0)


def test_actual_dataset_has_no_holdout_or_source_overlap():
    if not (META / "annotations.json").exists():
        pytest.skip("Local reviewed experiment dataset is not present")
    data = checked_manifest()  # Includes image hashes, source leakage, configured polygon.
    train, dev = selected(data["items"], "train"), selected(data["items"], "dev")
    assert len(train) == 15
    assert len(dev) == 13
    val = read_json(META / "validation.json")["items"]
    assert len(val) == 42
    assert sum(i["subset"] == "six_counter_without_hand" for i in val) == 6
    assert sum(i["subset"] == "empty_counter_28" for i in val) == 28
