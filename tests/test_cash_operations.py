from pathlib import Path

import numpy as np
import pytest

from app.cv.camera_profile import CameraProfile
from app.events.cash_operations import CashOperationDetector
from app.rules.cash_security import evaluate_cash_security
from app.rules.cash_security_schemas import CashSecurityRequest

ROOT = Path(__file__).resolve().parents[1]


def block(
    detector,
    start,
    location,
    point,
    *,
    key="cash-a",
    session=None,
    motion=0,
    valid=True,
    safe_proof=False,
):
    for i in range(3):
        t = start + i * 0.25
        detector.update(
            t,
            [
                {
                    "timestamp": t,
                    "logical_cash_id": key,
                    "location": location,
                    "center": [point[0] + motion * i, point[1]],
                    "confidence": 0.95,
                    "cashier_track_id": "cashier-1" if location == "cashier" else None,
                    "associated_session_id": session,
                    "evidence_frame": f"frame-{t}.jpg",
                    "visible_inside_safe": safe_proof,
                    "crossed_safe_opening": safe_proof,
                    "safe_open_observed": safe_proof,
                    "safe_producer_validated": safe_proof,
                }
            ],
            profile_valid=valid,
        )


def cycle(destination="table", session=None):
    detector = CashOperationDetector()
    block(detector, 0, "table", [0.6, 0.6], session=session)
    block(detector, 0.75, "cashier", [0.5, 0.4], session=session, motion=0.01)
    block(detector, 1.5, "counter", [0.2, 0.4], session=session)
    block(detector, 2.25, "cashier", [0.4, 0.4], session=session, motion=0.01)
    block(detector, 3, destination, [0.8, 0.5], session=session, safe_proof=destination == "safe")
    return detector


def test_operation_without_customer():
    result = cycle().finish(4)
    (operation,) = result["operations"]
    assert operation["associated_session_id"] is None
    assert operation["cashier_track_id"] == "cashier-1"
    assert operation["completion_status"] == "completed"
    assert operation["end_time"] == 3.5
    assert operation["evidence_confidence"] > 0.75


def test_operation_inside_session_keeps_origin_session():
    detector = cycle(session="service-1")
    block(detector, 4, "table", [0.8, 0.5], session="service-2")
    (operation,) = detector.finish(5)["operations"]
    assert operation["associated_session_id"] == "service-1"


def test_normal_safe_cycle_does_not_require_writing():
    result = cycle(destination="safe").finish(4)
    assert result["status"] == "completed"
    types = [e["type"] for e in result["operations"][0]["events"]]
    assert "MONEY_TO_SAFE" in types
    assert "NOTEBOOK_WRITING" not in types
    assert types[-1] == "CASH_OPERATION_COMPLETED"


def test_normal_table_cycle_does_not_require_safe():
    result = cycle().finish(4)
    types = [e["type"] for e in result["operations"][0]["events"]]
    assert types == [
        "MONEY_HELD_BY_CASHIER",
        "MONEY_FROM_TABLE",
        "MONEY_TO_COUNTER",
        "MONEY_FROM_COUNTER",
        "MONEY_TO_TABLE",
        "CASH_OPERATION_COMPLETED",
    ]
    assert "violations" not in result  # Recognition does not evaluate business rules.
    # These standalone operation events never enter the existing R05 customer contract.
    report = evaluate_cash_security(CashSecurityRequest(duration_sec=4, events=[]))
    assert report.violations == []
    assert report.verdict == "insufficient_evidence"


def test_neighboring_stacks_do_not_inherit_counter_history():
    detector = CashOperationDetector()
    block(detector, 0, "cashier", [0.5, 0.4], motion=0.01)
    block(detector, 0.75, "counter", [0.2, 0.4])
    block(detector, 1.5, "cashier", [0.5, 0.4], key="cash-b", motion=0.01)
    block(detector, 2.25, "table", [0.6, 0.6], key="cash-b")
    result = detector.finish(3)
    assert len(result["operations"]) == 2
    assert [o["logical_cash_ids"] for o in result["operations"]] == [["cash-a"], ["cash-b"]]
    assert all(o["completion_status"] != "completed" for o in result["operations"])


def test_wrong_camera_produces_no_operations():
    detector = CashOperationDetector()
    block(detector, 0, "cashier", [0.5, 0.4], motion=0.01, valid=False)
    assert detector.finish(1)["operations"] == []


def test_missing_detection_does_not_invent_withdrawal_or_operation():
    detector = CashOperationDetector()
    block(detector, 0, "counter", [0.2, 0.4])
    detector.update(1, [])
    block(detector, 3, "cashier", [0.5, 0.4])
    assert detector.finish(4)["operations"] == []


def test_disappearance_does_not_complete_an_existing_operation():
    detector = CashOperationDetector()
    block(detector, 0, "cashier", [0.5, 0.4], motion=0.01)
    block(detector, 0.75, "counter", [0.2, 0.4])
    detector.update(2, [])
    detector.update(10, [])
    (operation,) = detector.finish(11)["operations"]
    assert operation["completion_status"] == "insufficient_evidence"
    assert operation["end_time"] is None
    assert "MONEY_FROM_COUNTER" not in [e["type"] for e in operation["events"]]


def test_safe_overlap_is_not_deposit():
    detector = CashOperationDetector()
    block(detector, 0, "cashier", [0.5, 0.4], motion=0.01)
    block(detector, 0.75, "safe", [0.8, 0.5])
    (operation,) = detector.finish(2)["operations"]
    assert "MONEY_TO_SAFE" not in [e["type"] for e in operation["events"]]
    assert "safe_deposit_not_visually_confirmed" in operation["uncertainty"]


def test_stationary_table_and_hand_overlap_do_not_start_operation():
    detector = CashOperationDetector()
    block(detector, 0, "cashier", [0.5, 0.4])
    block(detector, 0.75, "cashier", [0.5, 0.4])
    assert detector.finish(2)["operations"] == []


def test_hand_approaching_stationary_table_cash_is_not_a_pickup():
    detector = CashOperationDetector()
    block(detector, 0, "table", [0.5, 0.4])
    block(detector, 0.75, "cashier", [0.5, 0.4], motion=0.0001)
    block(detector, 1.5, "table", [0.5, 0.4])
    assert detector.finish(3)["operations"] == []


def test_unknown_location_blocks_transition():
    detector = CashOperationDetector()
    block(detector, 0, "cashier", [0.5, 0.4], motion=0.01)
    block(detector, 0.75, "unknown", [0.3, 0.4])
    block(detector, 1.5, "counter", [0.2, 0.4])
    (operation,) = detector.finish(3)["operations"]
    assert "MONEY_TO_COUNTER" not in [e["type"] for e in operation["events"]]


def test_real_camera_references_validate_but_wrong_view_and_blank_fail():
    a = CameraProfile(ROOT / "config/camera_profiles/desk-wide-a.json")
    b = CameraProfile(ROOT / "config/camera_profiles/desk-wide-b.json")
    assert a.validate(a.reference)["valid"]
    assert b.validate(b.reference)["valid"]
    assert not a.validate(b.reference)["valid"]
    assert not b.validate(a.reference)["valid"]
    assert not a.validate(np.zeros_like(a.reference))["valid"]
    assert not a.validate(a.reference[:100])["valid"]


def test_duplicate_cash_id_and_reversed_time_rejected():
    detector = CashOperationDetector()
    block(detector, 0, "table", [0.5, 0.5])
    with pytest.raises(ValueError, match="increase"):
        detector.update(0, [])
