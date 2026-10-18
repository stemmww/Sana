"""R05 synthetic temporal evidence, without modifying legacy scenario fixtures."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.events.cash_security import CashSecurityEventDetector
from app.main import create_app
from app.rules.cash_security import evaluate_cash_security
from app.rules.cash_security_schemas import CashSecurityRequest


def event(kind, t, **kw):
    value = {
        "event_id": f"{kind}-{t}",
        "type": kind,
        "start_sec": t,
        "end_sec": t + 0.5,
        "source": "synthetic_example",
        "status": "confirmed",
        "sample_count": 3,
        "confidence": 0.98,
        "customer_id": "customer-a",
        "operation_id": "operation-a",
        "person_id": "cashier-a",
        "object_id": "id-a" if kind.startswith("ID_") else "cash-a",
        "money_group_id": "deposit-a",
        "safe_id": "safe-a",
        "evidence_times": [t, t + 0.25, t + 0.5],
        "evidence_frames": ["synthetic-frame-1", "synthetic-frame-2", "synthetic-frame-3"],
    }
    value.update(kw)
    return value


def base():
    return [
        event(k, t)
        for t, k in enumerate(
            [
                "CASHIER_PRESENT",
                "SERVICE_STARTED",
                "ID_TRANSFER_TO_CASHIER",
                "ID_CHECK",
                "ID_ON_TABLE",
                "MONEY_TRANSFER_TO_CASHIER",
                "MONEY_TO_COUNTER",
                "MONEY_FROM_COUNTER",
                "MONEY_HELD_BY_CASHIER",
            ]
        )
    ]


def boundary(kind="CUSTOMER_ENTER", customer="customer-b", operation="operation-b"):
    return event(kind, 16, customer_id=customer, operation_id=operation)


def table(start=9, end=17, **kw):
    times = [start + 0.25 * i for i in range(round((end - start) / 0.25) + 1)]
    return event(
        "MONEY_LEFT_ON_TABLE",
        start,
        end_sec=end,
        sample_count=len(times),
        evidence_times=times,
        **kw,
    )


def run(events):
    return evaluate_cash_security(
        CashSecurityRequest(
            duration_sec=max(e["end_sec"] for e in events) + 1,
            events=sorted(events, key=lambda e: e["end_sec"]),
        )
    )


def test_r05_old_money_at_next_customer_boundary_and_exact_time():
    report = run(base() + [event("CUSTOMER_EXIT", 14), boundary(), table()])
    assert report.verdict == "violation"
    alarm = report.violations[0]
    assert alarm["rule_id"] == "R05"
    assert alarm["timecode_sec"] == 16.5
    assert alarm["detected_at_sec"] == 17
    assert alarm["customer_id"] == "customer-a"
    assert alarm["next_customer_id"] == "customer-b"


@pytest.mark.parametrize(
    "change",
    [
        "temporary",
        "wrong_group",
        "wrong_customer",
        "uncertain_table",
        "uncertain_withdrawal",
        "no_withdrawal",
        "no_boundary",
        "sparse_table",
        "unknown_origin",
    ],
)
def test_no_false_alarm_from_incomplete_or_unrelated_evidence(change):
    events = base() + [boundary(), table()]
    if change == "temporary":
        events[-1] = table(9, 10)
    elif change == "wrong_group":
        events[-1]["money_group_id"] = "other-money"
    elif change == "wrong_customer":
        events[-1]["customer_id"] = "customer-b"
    elif change == "uncertain_table":
        events[-1].update(status="uncertain", uncertainty=["hand occlusion"])
    elif change == "uncertain_withdrawal":
        events[7].update(status="uncertain", uncertainty=["wrist hypothesis only"])
    elif change == "no_withdrawal":
        events = [e for e in events if e["type"] != "MONEY_FROM_COUNTER"]
    elif change == "no_boundary":
        events = [e for e in events if e["type"] != "CUSTOMER_ENTER"]
    elif change == "sparse_table":
        events[-1]["evidence_times"] = [9, 17]
    elif change == "unknown_origin":
        events[-1].update(
            status="insufficient_evidence",
            operation_id=None,
            money_group_id=None,
            uncertainty=["unknown origin"],
        )
    report = run(events)
    assert not report.violations
    assert report.verdict == "insufficient_evidence"


def test_cash_in_safe_before_next_customer_is_not_r05():
    report = run(
        base() + [table(9, 10), event("SAFE_OPENED", 11), event("MONEY_TO_SAFE", 12), boundary()]
    )
    assert report.verdict == "no_violation"
    assert not report.violations


def test_missing_safe_event_alone_is_not_a_violation():
    assert run(base() + [boundary()]).verdict == "insufficient_evidence"


def test_later_secure_does_not_erase_violation_at_boundary():
    assert run(base() + [boundary(), table(), event("MONEY_TO_SAFE", 18)]).verdict == "violation"


def test_conflicting_same_money_in_safe_and_on_table_requires_review():
    report = run(base() + [event("MONEY_TO_SAFE", 12), boundary(), table()])
    assert report.verdict == "insufficient_evidence"
    assert report.boundary_checks[0]["reason"] == "contradictory_safe_and_table_identity"


def test_new_service_triggers_once_even_with_entry_and_repeated_presence():
    events = base() + [
        boundary(),
        boundary("SERVICE_STARTED"),
        event("CUSTOMER_PRESENT", 17, customer_id="customer-b"),
        table(),
    ]
    assert len(run(events).violations) == 1
    assert (
        run(base() + [boundary("SERVICE_STARTED", customer="customer-a"), table()]).verdict
        == "violation"
    )


def test_full_normal_workflow_and_reset_for_next_operation():
    events = (
        base()
        + [
            event(kind, t)
            for t, kind in enumerate(
                [
                    "SAFE_OPENED",
                    "MONEY_TO_SAFE",
                    "SAFE_CLOSED",
                    "SAFE_LOCKED_VISUAL",
                    "ID_RETURNED_TO_CUSTOMER",
                    "CUSTOMER_EXIT",
                ],
                start=9,
            )
        ]
        + [boundary("SERVICE_STARTED")]
    )
    report = run(events)
    assert report.verdict == "no_violation"
    assert report.operations[0]["status"] == "completed"
    assert report.operations[1]["state"] == "ID_TRANSFER_TO_CASHIER"
    assert report.operations[1]["accepted_steps"] == []
    events[12]["status"] = "uncertain"
    events[12]["uncertainty"] = ["key too small"]
    assert run(events).operations[0]["status"] == "insufficient_evidence"


def test_absolute_times_and_ids_are_not_scenario_labels():
    events = base() + [boundary(), table()]
    for e in events:
        e["start_sec"] += 100
        e["end_sec"] += 100
        e["evidence_times"] = [t + 100 for t in e["evidence_times"]]
        e["event_id"] = "arbitrary-" + e["event_id"]
    report = run(events)
    assert report.verdict == "violation"
    assert report.violations[0]["timecode_sec"] == 116.5


def detector():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "config/cash_security_camera.json").read_text(
            encoding="utf-8"
        )
    )
    return CashSecurityEventDetector(config)


def test_safe_motion_disappearance_and_closed_door_do_not_prove_actions():
    d = detector()
    for kind in ["SAFE_OPENED", "MONEY_TO_SAFE", "SAFE_CLOSED", "SAFE_LOCKED_VISUAL"]:
        result = d.observe(
            {
                "event": event(kind, 5),
                "facts": ["region_motion"],
                "association_reliable": True,
                "producer_validated": False,
                "producer": "local_visual",
            }
        )
        assert result.status == "insufficient_evidence"


def test_positive_table_requires_outside_safe_and_hand_clear():
    d = detector()
    observation = {
        "event": event("MONEY_LEFT_ON_TABLE", 5),
        "facts": ["cash_visible", "stationary_on_table", "hand_clear"],
        "object_centers": [[0.40, 0.52]] * 3,
        "association_reliable": True,
        "producer_validated": True,
        "producer": "synthetic_example",
    }
    assert d.observe(observation).status == "confirmed"
    observation["object_centers"] = [[0.75, 0.5]] * 3
    assert d.observe(observation).status == "insufficient_evidence"


def test_api_authentication_and_separate_route(tmp_path):
    settings = Settings(_env_file=None, api_key="security-test-key", storage_dir=tmp_path)
    payload = {
        "duration_sec": 20,
        "events": sorted(base() + [boundary(), table()], key=lambda e: e["end_sec"]),
    }
    with TestClient(create_app(settings)) as client:
        route = "/api/v1/rules/cash-security/evaluate"
        assert client.post(route, json=payload).status_code == 401
        response = client.post(route, json=payload, headers={"X-API-Key": "security-test-key"})
        assert response.status_code == 200
        assert response.json()["violations"][0]["rule_id"] == "R05"
        assert "/api/v1/rules/evaluate" in client.get("/openapi.json").json()["paths"]


def test_uncertain_additional_withdrawal_prevents_global_no_violation():
    events = base() + [event("MONEY_TO_SAFE", 12), boundary()]
    events.append(
        event(
            "MONEY_FROM_COUNTER",
            18,
            operation_id="operation-b",
            customer_id="customer-b",
            money_group_id="deposit-b",
            status="uncertain",
            uncertainty=["occluded money"],
        )
    )
    assert run(events).verdict == "insufficient_evidence"


def test_returned_id_requires_same_card_and_observed_original_transfer():
    from app.rules.cash_security_schemas import SecurityEvent

    d = detector()
    obs = {
        "event": event("ID_RETURNED_TO_CUSTOMER", 15),
        "facts": ["id_visible", "cashier_to_customer_transition", "same_id_confirmed"],
        "association_reliable": True,
        "producer_validated": True,
        "producer": "synthetic_example",
    }
    assert d.observe(obs).status == "insufficient_evidence"
    d.events.append(SecurityEvent(**event("ID_TRANSFER_TO_CASHIER", 2)))
    assert d.observe(obs).status == "confirmed"
    obs["event"]["object_id"] = "different-card"
    assert d.observe(obs).status == "insufficient_evidence"


def test_positive_safe_actions_require_observed_open_close_and_lock_action():
    from app.events.cash_security import REQUIRED_FACTS

    d = detector()
    for t, kind in enumerate(
        ["SAFE_OPENED", "MONEY_TO_SAFE", "SAFE_CLOSED", "SAFE_LOCKED_VISUAL"], start=9
    ):
        result = d.observe(
            {
                "event": event(kind, t),
                "facts": REQUIRED_FACTS[kind],
                "association_reliable": True,
                "producer_validated": True,
                "producer": "synthetic_example",
            }
        )
        assert result.status == "confirmed"
    assert "physical locking is not established" in result.observation_basis
