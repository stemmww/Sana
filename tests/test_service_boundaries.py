from app.events.config import ServiceBoundaryConfig
from tests.event_fixtures import Scenario, person, rectangle


def scene():
    return Scenario(
        service_boundary=ServiceBoundaryConfig(
            anchor_height=0.5,
            zones={
                "interior": rectangle(0.2, 0.2, 0.5, 0.9),
                "portal": rectangle(0.1, 0.2, 0.19, 0.9),
                "exterior": rectangle(0, 0.2, 0.09, 0.9),
            },
        )
    ).start()


def at(s, x, key="client", frames=4):
    s.people[1] = person(key, point=(x, 0.5))
    s.hold(frames=frames)


def leave(s):
    at(s, 0.15)
    at(s, 0.065)


def test_one_customer_confirmed_exit_one_service():
    s = scene()
    leave(s)
    sessions = s.finish()["rule_result"]["sessions"]
    assert len(sessions) == 1 and sessions[0]["end_sec"] is not None
    assert len(s.detector.service_boundary.transitions) == 1


def test_sequential_customer_reusing_raw_id_gets_new_session_without_r02():
    s = scene()
    leave(s)
    at(s, 0.15)
    at(s, 0.25)
    doc = s.finish()
    sessions = doc["rule_result"]["sessions"]
    assert len(sessions) == 2
    assert sessions[0]["customer_id"] != sessions[1]["customer_id"]
    assert sessions[0]["end_sec"] < sessions[1]["start_sec"]
    assert not [v for v in doc["rule_result"]["violations"] if v["rule_id"] == "R02"]
    assert s.detector.previous_frame.people[1].track_id == "client"


def test_new_raw_id_is_fragment_not_new_customer():
    s = scene()
    before = s.detector.active_customer
    at(s, 0.26, "fragment")
    assert s.detector.active_customer == before
    assert len(s.finish()["rule_result"]["sessions"]) == 1
    assert s.detector.service_boundary.track_changes[-1]["from_track_id"] == "client"


def test_detection_gap_does_not_end_service_or_lose_cashier():
    s = scene()
    saved = s.people
    s.people = []
    s.hold(frames=3)
    s.people = saved
    s.hold(frames=4)
    doc = s.finish()
    assert len(doc["rule_result"]["sessions"]) == 1
    assert doc["rule_result"]["sessions"][0]["end_sec"] is None
    assert "CASHIER_EXIT" not in s.kinds()
    assert s.detector.cashier_id == "cashier"


def test_simultaneous_distinct_customers_keep_r02():
    s = scene()
    s.people.append(person("second", point=(0.42, 0.5)))
    s.hold(frames=4)
    assert [v["rule_id"] for v in s.finish()["rule_result"]["violations"]] == ["R02"]


def test_door_approach_and_return_is_not_exit():
    s = scene()
    at(s, 0.15)
    at(s, 0.25)
    assert "CUSTOMER_EXIT" not in s.kinds()
    assert len(s.finish()["rule_result"]["sessions"]) == 1


def test_absence_after_door_approach_is_not_exit():
    s = scene()
    at(s, 0.15)
    s.people.pop()
    s.hold(frames=20)
    assert "CUSTOMER_EXIT" not in s.kinds()


def test_far_raw_id_reuse_without_observed_exit_remains_unknown():
    s = scene()
    at(s, 0.48, frames=4)
    assert len(s.finish()["rule_result"]["sessions"]) == 1
    assert not s.detector.service_boundary.transitions
    assert s.detector.previous_frame.people[1].identity_reliable is False
    assert s.detector.service_boundary.uncertainties
