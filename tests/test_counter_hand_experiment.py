"""Safety properties of the independent wrist diagnostic, using synthetic observations."""

import copy

from experiments.counter_hand import analyze, observations
from experiments.counter_hand_data import META
from experiments.counter_occupancy import read_json


def scenario(owner="cashier"):
    frames = []
    for t, x, d in [
        (0, 200, 0.8),
        (0.1, 170, 0.5),
        (0.2, 120, 0),
        (0.3, 120, 0),
        (0.4, 150, 0.25),
        (0.5, 180, 0.6),
        (0.7, 180, 0.6),
        (0.9, 180, 0.6),
    ]:
        hand = {
            "timestamp_sec": t,
            "frame_index": round(t * 30),
            "track_id": "hand-1",
            "side": "left",
            "xy": [x, 100],
            "distance_widths": d,
            "core_distance_widths": d,
            "inside_output_zone": d == 0,
            "owner": owner,
            "wrist_confidence": 0.9,
        }
        frames.append(
            {
                "timestamp_sec": t,
                "frame_index": hand["frame_index"],
                "counter_width_px": 100,
                "hands": [hand],
            }
        )
    cash = []
    for t, identity in [(0.5, "cash-a"), (0.7, "cash-b")]:
        cash.append(
            {
                "timestamp_sec": t,
                "frame_index": round(t * 30),
                "width": 400,
                "height": 400,
                "objects": [
                    {
                        "label": "cash",
                        "confidence": 0.8,
                        "center": [0.45, 0.25],
                        "bbox": [0.43, 0.23, 0.47, 0.27],
                        "logical_object_id": identity,
                    }
                ],
            }
        )
    return frames, cash, read_json(META / "protocol.json")


def test_cash_track_fragments_do_not_block_supported_sequence():
    f, c, p = scenario()
    result = analyze(f, c, p)
    assert result["summary"]["withdrawal_hypotheses"] == 1
    assert {m["type"] for m in result["markers"]} == {
        "HAND_APPROACH",
        "HAND_IN_OUTPUT_ZONE",
        "HAND_LEAVE_OUTPUT_ZONE",
    }
    assert len({n["cash_logical_id"] for n in result["contacts"][0]["cash_links"]}) == 2


def test_empty_touch_is_not_a_withdrawal():
    f, _, p = scenario()
    result = analyze(f, [], p)
    assert result["summary"]["complete_sequences"] == 1
    assert result["summary"]["withdrawal_hypotheses"] == 0


def test_unknown_owner_stays_unknown_despite_cash():
    f, c, p = scenario("unknown")
    result = analyze(f, c, p)
    assert result["summary"]["unknown_owner_contacts"] == 1
    assert result["summary"]["withdrawal_hypotheses"] == 0


def test_disappearance_and_clip_end_are_not_hand_leave():
    f, c, p = scenario()
    result = analyze(f[:4], c, p)
    assert result["summary"]["output_contacts"] == 1
    assert result["summary"]["complete_sequences"] == 0
    assert result["summary"]["withdrawal_hypotheses"] == 0


def test_cash_already_in_hand_is_not_new_extraction():
    f, c, p = scenario()
    before = copy.deepcopy(c[0])
    before.update(timestamp_sec=0.1, frame_index=3)
    c.insert(0, before)
    assert analyze(f, c, p)["summary"]["withdrawal_hypotheses"] == 0


def test_relative_time_and_arbitrary_metadata_do_not_change_decisions():
    f, c, p = scenario()
    original = analyze(f, c, p)
    for frame in f + c:
        frame["timestamp_sec"] += 123.0
        frame["video_name"] = "arbitrary-negative-or-positive-name"
        for hand in frame.get("hands", []):
            hand["timestamp_sec"] += 123.0
    shifted = analyze(f, c, p)
    assert shifted["summary"] == original["summary"]
    assert shifted["contacts"][0]["withdrawal_hypothesis"]


def test_torso_and_arm_allow_wrist_outside_person_bbox():
    p = read_json(META / "protocol.json")
    pose = {
        "bbox_px": [15, 20, 90, 90],
        "confidence": 0.9,
        "joints": {str(k): [65, 45 if k < 11 else 65, 0.9] for k in range(5, 13)},
    }
    pose["joints"]["9"] = [20, 50, 0.9]
    pose["joints"]["10"] = [80, 50, 0.9]
    raw = [
        {"timestamp_sec": t, "frame_index": i, "width": 100, "height": 100, "poses": [pose]}
        for i, t in enumerate([0, 0.067, 0.134])
    ]
    roles = [
        {
            "timestamp_sec": 0,
            "people": [
                {
                    "bbox": [0.4, 0.2, 0.9, 0.9],
                    "track_id": "person-a",
                    "role": "cashier",
                    "identity_reliable": True,
                }
            ],
        }
    ]
    result = observations(raw, roles, p)
    assert result[0]["hands"][0]["owner"] == "unknown"
    assert result[-1]["hands"][0]["owner"] == "cashier"
    assert result[-1]["hands"][0]["xy"][0] < 40
    roles[0]["people"].append({**roles[0]["people"][0], "track_id": "person-b", "role": "customer"})
    ambiguous = observations(raw, roles, p)
    assert all(h["owner"] == "unknown" for f in ambiguous for h in f["hands"])
