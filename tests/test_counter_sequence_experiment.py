"""Safety and invariance checks for the isolated sequence diagnostic."""

import copy

import pytest

from experiments.counter_occupancy import read_json
from experiments.counter_sequence import TYPES, analyze
from experiments.counter_sequence_data import DATA, META
from experiments.counter_sequence_run import score


@pytest.fixture
def sample():
    if not (DATA / "x1/features.json").exists():
        pytest.skip("Local diagnostic cache is not installed")
    return read_json(DATA / "x1/features.json")["frames"], read_json(META / "protocol.json")


def signature(result):
    return [
        (e["type"], e.get("hypothesis"), e["timestamp_sec"], e["confidence"])
        for e in result["events"]
    ]


def test_disappearance_alone_never_proves_counter_interaction(sample):
    frames, p = sample
    for frame in frames:
        frame["cashier_reach_motion_proxy"] = False
        frame["counter_motion_fraction"] = 0
    types = [e["type"] for e in analyze(frames, p)["events"]]
    assert "CASH_DISAPPEARED_NEAR_COUNTER" in types
    assert "PROBABLE_COUNTER_INTERACTION" not in types


def test_motion_without_cash_cannot_create_interaction(sample):
    frames, p = sample
    for frame in frames:
        frame["objects"] = []
        frame["cashier_reach_motion_proxy"] = True
    assert not analyze(frames, p)["events"]


def test_distant_cash_disappearance_is_not_counter_interaction(sample):
    frames, p = sample
    for frame in frames:
        for obj in frame["objects"]:
            obj["counter_distance_widths"] = 5
    assert not analyze(frames, p)["events"]


def test_filenames_absolute_times_and_inside_cash_do_not_decide(sample):
    frames, p = sample
    baseline = analyze(frames, p)
    changed = copy.deepcopy(frames)
    for frame in changed:
        frame["timestamp_sec"] += 123
        frame["video"] = "unseen-arbitrary-name.mov"
        # Even strong continuous detections inside the machine cannot supply sequence evidence.
        frame["objects"].append({"label": "cash", "confidence": 1.0, "inside_counter": True})
    got = analyze(changed, p)
    assert len(got["events"]) == len(baseline["events"])
    for a, b in zip(baseline["events"], got["events"], strict=True):
        assert (a["type"], a.get("hypothesis"), a["confidence"]) == (
            b["type"],
            b.get("hypothesis"),
            b["confidence"],
        )
        assert b["timestamp_sec"] - 123 == pytest.approx(a["timestamp_sec"])


def test_no_business_events_or_missing_uncertainty(sample):
    frames, p = sample
    for e in analyze(frames, p)["events"]:
        assert e["type"] in TYPES
        assert not e["production_event"]
        assert e["evidence_frames"] and e["observed_features"] and e["uncertainty_reasons"]


def test_false_cash_reappearance_does_not_become_withdrawal():
    if not (DATA / "x3/features.json").exists():
        pytest.skip("Local diagnostic cache is not installed")
    frames = read_json(DATA / "x3/features.json")["frames"]
    result = analyze(frames, read_json(META / "protocol.json"))
    assert any(e["type"] == "CASH_REAPPEARED_NEAR_COUNTER" for e in result["events"])
    assert not any(e.get("hypothesis") == "possible_withdrawal" for e in result["events"])


def test_duplicate_hypotheses_count_as_false_positive():
    event = {
        "id": "one",
        "type": "PROBABLE_COUNTER_INTERACTION",
        "hypothesis": "possible_deposit",
        "timestamp_sec": 10,
        "confidence": 0.55,
    }
    truth = {
        "actions": [
            {"id": "deposit", "hypothesis": "possible_deposit", "start_sec": 9, "end_sec": 11}
        ],
        "negative_intervals": [[0, 8]],
    }
    metrics = score([event, {**event, "id": "two"}], truth)
    assert metrics["detected"] == 1
    assert metrics["false_events"] == 1
