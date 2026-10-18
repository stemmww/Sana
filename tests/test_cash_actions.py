"""Safety properties of the offline experiment; synthetic data never enters real runs."""

import copy

from experiments.cash_actions import grade, wrist_cash


def parameters():
    return {
        "cash_pose_time_tolerance_sec": 0.1,
        "cash_confidence": 0.25,
        "cash_wrist_distance_widths": 0.6,
        "cash_owner_margin_widths": 0.1,
        "cash_min_samples": 2,
        "roi_change_fraction": 0.05,
    }


def observed(owner="cashier", body="person-1", second_owner="cashier", second_body="person-1"):
    hands = [
        {
            "timestamp_sec": 1.0,
            "width": 100,
            "height": 100,
            "counter_width_px": 100,
            "hands": [
                {
                    "xy": [48, 50],
                    "track_id": "left",
                    "owner": owner,
                    "body_track_id": body,
                    "wrist_confidence": 0.9,
                },
                {
                    "xy": [52, 50],
                    "track_id": "right",
                    "owner": second_owner,
                    "body_track_id": second_body,
                    "wrist_confidence": 0.9,
                },
            ],
        }
    ]
    cash = [
        {
            "timestamp_sec": 1.0,
            "frame_index": 30,
            "objects": [
                {
                    "label": "cash",
                    "confidence": 0.8,
                    "bbox": [0.45, 0.45, 0.55, 0.55],
                    "logical_object_id": "cash-a",
                }
            ],
        }
    ]
    return hands, cash


def proposal(kind="CASH_EXTRACTION_ACTION"):
    return {
        "type": kind,
        "owner": "cashier",
        "features": {
            "approach": True,
            "leave": True,
            "contact_inside_window": True,
            "cash_before_samples": 0,
            "cash_after_samples": 2,
        },
        "roi": {"raw_during_peak": 0.3},
        "observations": {"cash_before": [], "cash_after": []},
    }


def test_two_hands_one_person_preserve_owner_not_wrist_certainty():
    hands, cash = observed()
    node = wrist_cash(hands, cash, parameters())[0]
    assert node["owner"] == "cashier"
    assert node["ambiguous"] and not node["owner_ambiguous"]
    assert set(node["competing_hand_ids"]) == {"left", "right"}


def test_different_people_remain_unknown():
    hands, cash = observed(second_owner="customer", second_body="person-2")
    node = wrist_cash(hands, cash, parameters())[0]
    assert node["owner"] == "unknown" and node["owner_ambiguous"]


def test_unknown_hand_cannot_become_known_via_neighbour():
    hands, cash = observed(second_owner="unknown", second_body=None)
    assert wrist_cash(hands, cash, parameters())[0]["owner"] == "unknown"


def test_cash_gap_does_not_reconnect_different_objects_or_mutate_inputs():
    hands, cash = observed()
    original = copy.deepcopy((hands, cash))
    a = wrist_cash(hands, cash, parameters())[0]
    cash[0]["objects"][0]["logical_object_id"] = "cash-b"
    b = wrist_cash(hands, cash, parameters())[0]
    assert a["logical_cash_id"] != b["logical_cash_id"]
    cash[0]["objects"][0]["logical_object_id"] = "cash-a"
    assert (hands, cash) == original


def test_hand_motion_and_roi_change_without_cash_is_not_action_hypothesis():
    candidate = proposal()
    candidate["features"]["cash_after_samples"] = 0
    assert grade(candidate, parameters())["status"] == "insufficient_evidence"
    existing_cash = proposal()
    existing_cash["features"]["cash_before_samples"] = 2
    assert grade(existing_cash, parameters())["status"] == "insufficient_evidence"


def test_ID_observation_cannot_supply_cash_support():
    hands, cash = observed()
    cash[0]["objects"][0]["label"] = "id_card"
    assert wrist_cash(hands, cash, parameters()) == []


def test_positive_multicue_sequence_remains_hypothesis_not_confirmed_event():
    result = grade(proposal(), parameters())
    assert result["status"] == "hypothesis"
    assert "independent_negative_action_validation_missing" in result["uncertainty"]


def test_no_visible_leave_or_unknown_owner_rejects_hypothesis():
    for key in ("leave", "owner"):
        candidate = proposal()
        if key == "leave":
            candidate["features"]["leave"] = False
        else:
            candidate["owner"] = "unknown"
        assert grade(candidate, parameters())["status"] == "insufficient_evidence"


def test_handover_requires_positive_recipient_cash_and_same_object():
    for after, ids, direction in [(0, ["cash-a"], True), (2, [], True), (2, ["cash-a"], False)]:
        candidate = proposal("CASH_HANDOVER_TO_CUSTOMER")
        candidate["features"].update(
            cash_before_samples=3,
            cash_after_samples=after,
            same_logical_cash_ids=ids,
            cash_moves_to_recipient=direction,
        )
        assert grade(candidate, parameters())["status"] == "insufficient_evidence"


def test_stale_pose_cannot_establish_cash_owner():
    hands, cash = observed()
    cash[0]["timestamp_sec"] = 1.5
    assert wrist_cash(hands, cash, parameters()) == []
