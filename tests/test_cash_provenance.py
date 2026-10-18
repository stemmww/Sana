from app.events.cash_provenance import analyze_cash_provenance, bind_cash_event
from tests.event_fixtures import Scenario, config
from tests.test_cash_security import event, run

SESSIONS = [
    {"session_id": "op1", "customer_id": "c1", "start_sec": 0, "end_sec": 3},
    {"session_id": "op2", "customer_id": "c2", "start_sec": 4, "end_sec": None},
]


def samples(key, times, x=0.5):
    return [
        {
            "timestamp_sec": t,
            "objects": [
                {
                    "logical_object_id": key,
                    "label": "cash",
                    "timestamp_sec": t,
                    "bbox": [x - 0.01, 0.79, x + 0.01, 0.81],
                    "center": [x, 0.8],
                    "confidence": 0.98,
                    "zone": "table",
                    "association": "table",
                    "association_ambiguous": False,
                }
            ],
        }
        for t in times
    ]


def origin():
    return event(
        "MONEY_TRANSFER_TO_CASHIER", 1, object_id="a", operation_id="op1", customer_id="c1"
    )


def audit(frames, events=None):
    return analyze_cash_provenance(
        sorted(frames, key=lambda f: f["timestamp_sec"]), SESSIONS, events or [origin()], config()
    )


def recovered():
    return audit(
        samples("a", [1, 1.25, 1.5, 2, 2.25, 2.5, 2.75])
        + samples("b", [3.5, 3.75, 4, 4.25, 4.5], 0.505)
    )


def test_short_fragment_gap_and_session_boundary_preserve_origin():
    p = recovered()
    assert len(p["objects"]) == 1
    obj = p["objects"][0]
    assert obj["source_session_id"] == "op1"
    assert obj["object_track_ids"] == ["a", "b"]
    assert obj["first_seen"] == 1 and obj["last_seen"] == 4.5
    assert obj["status"] == "confirmed"
    assert len(obj["table_presence_history"]) == 12


def test_new_stack_does_not_inherit_just_because_it_is_on_table():
    p = audit(samples("a", [1, 1.25, 1.5]) + samples("b", [2, 2.25, 2.5], 0.7))
    assert len(p["objects"]) == 2
    assert p["objects"][1]["status"] == "insufficient_evidence"


def test_long_gap_even_same_position_is_not_connected():
    p = audit(samples("a", [1, 1.25, 1.5]) + samples("b", [4, 4.25, 4.5]))
    assert len(p["objects"]) == 2
    assert p["objects"][1]["source_session_id"] == "op2"


def test_simultaneous_bundles_cannot_merge():
    p = audit(samples("a", [1, 1.25, 1.5, 1.75, 2]) + samples("b", [1.5, 1.75, 2], 0.505))
    assert len(p["objects"]) == 2


def test_two_plausible_predecessors_remain_insufficient():
    p = audit(
        samples("a", [1, 1.25, 1.5], 0.49)
        + samples("b", [1, 1.25, 1.5], 0.51)
        + samples("c", [2, 2.25, 2.5])
    )
    assert len(p["objects"]) == 3
    assert "multiple_plausible_cash_predecessors" in p["objects"][-1]["uncertainty"]


def test_cash_first_seen_between_sessions_cannot_be_assigned_to_next():
    p = audit(samples("a", [3.25, 3.5, 4, 4.25]))
    assert p["objects"][0]["source_session_id"] is None


def r05_events(p, secure=False):
    removed = event("MONEY_FROM_COUNTER", 2, object_id="a", operation_id="op1", customer_id="c1")
    old_exit = event("CUSTOMER_EXIT", 2.5, operation_id="op1", customer_id="c1")
    next_service = event("SERVICE_STARTED", 3.5, operation_id="op2", customer_id="c2")
    table = event(
        "MONEY_LEFT_ON_TABLE",
        3.5,
        end_sec=4.5,
        sample_count=5,
        evidence_times=[3.5, 3.75, 4, 4.25, 4.5],
        object_id="b",
    )
    events = [bind_cash_event(removed, p), old_exit, next_service]
    if secure:
        events.append(bind_cash_event(event("MONEY_TO_SAFE", 2.5, object_id="a"), p))
    else:
        events.append(bind_cash_event(table, p))
    return events


def test_retained_previous_session_cash_reaches_unchanged_r05():
    result = run(r05_events(recovered()))
    assert [v["rule_id"] for v in result.violations] == ["R05"]
    assert result.violations[0]["operation_id"] == "op1"
    assert result.violations[0]["timecode_sec"] == 4


def test_cash_secured_before_next_service_does_not_trigger_r05():
    assert not run(r05_events(recovered(), secure=True)).violations


def test_uncertain_origin_cannot_trigger_r05():
    p = recovered()
    p["objects"][0].update(
        status="insufficient_evidence", provenance_confidence=0.5, uncertainty=["ambiguous_origin"]
    )
    report = run(r05_events(p))
    assert report.verdict == "insufficient_evidence" and not report.violations


def test_secured_bundle_is_not_recovered_as_new_cash_on_table():
    secured = event("MONEY_TO_SAFE", 1.5, object_id="a")
    p = audit(samples("a", [1, 1.25, 1.5]) + samples("b", [2.25, 2.5, 2.75]), [origin(), secured])
    assert len(p["objects"]) == 2


def test_legacy_cash_state_is_not_reset_when_service_changes():
    s = Scenario(cash_provenance_enabled=True, exit_requires_boundary=False).start()
    s.hold((0.5, 0.85))
    key = next(iter(s.detector.objects))
    group = s.detector.objects[key].group_id
    s.people.pop()
    s.hold((0.5, 0.85), frames=12)
    assert s.detector.active_session is None
    assert s.detector.objects[key].group_id == group
    assert s.detector.history[-1]["source_session_id"] == "service-1"
