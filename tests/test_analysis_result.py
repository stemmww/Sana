"""Recognition completeness must not create or suppress Rule Engine violations."""

import base64
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.events.cash_operations import CashOperationDetector
from app.main import create_app
from app.rules.cash_security import evaluate_cash_security
from app.rules.cash_security_schemas import CashSecurityRequest
from app.rules.engine import evaluate
from app.rules.schemas import RuleRequest
from app.schemas import AnalysisResult, EvidenceFrame
from app.services.analysis_result import compose_analysis_result
from tests.test_cash_operations import block, cycle
from tests.test_cash_security import base, boundary, event, run, table
from tests.test_jobs import KEY, read_job, upload
from tests.test_sequence_rules import event as legacy_event
from tests.test_sequence_rules import result as legacy_result
from tests.test_sequence_rules import timeline


def incomplete():
    detector = CashOperationDetector()
    block(detector, 0, "cashier", [0.5, 0.4], motion=0.01)
    return detector.finish(4)


def reports():
    return {
        "R01-R04": evaluate(RuleRequest(duration_sec=30)),
        "R05": evaluate_cash_security(CashSecurityRequest(duration_sec=30)),
    }


def evidence(alarm):
    # Transport fixture only. Production obtains actual JPEGs from the source video.
    return [
        EvidenceFrame(
            timestamp_sec=alarm["timecode_sec"],
            image_base64=base64.b64encode(b"\xff\xd8\xfftest\xff\xd9").decode(),
        )
    ]


def test_incomplete_operation_without_alarm_is_no_violation_and_keeps_diagnostics():
    cash = incomplete()
    original = deepcopy(cash)
    value = compose_analysis_result(reports(), cash)
    assert value.verdict == "no_violation" and value.violations == []
    assert value.diagnostics["cash_operation_status"] == "insufficient_evidence"
    assert value.diagnostics["analysis_status"] == "incomplete"
    assert value.diagnostics["cash_operation"]["operations"][0]["end_time"] is None
    assert value.diagnostics["insufficient_evidence_reasons"]
    assert value.diagnostics["full_procedural_compliance"] == "not_established"
    assert cash == original


def test_completed_operation_without_alarm_is_no_violation():
    value = compose_analysis_result(reports(), cycle().finish(4))
    assert value.verdict == "no_violation"
    assert value.diagnostics["cash_operation_status"] == "completed"
    # Rule evidence is still incomplete, independently of operation recognition.
    assert value.diagnostics["rule_reports"]["R05"]["verdict"] == "insufficient_evidence"
    assert value.diagnostics["analysis_status"] == "incomplete"


def test_incomplete_operation_does_not_hide_confirmed_legacy_violation():
    rule_reports = reports()
    rule_reports["R01-R04"] = legacy_result(
        timeline(
            [
                legacy_event("CASHIER_PRESENT", 0),
                legacy_event("CUSTOMER_ENTER", 1),
                legacy_event("MONEY_TRANSFER_TO_CASHIER", 2),
                legacy_event("MONEY_TO_COUNTER", 3),
            ]
        )
    )
    value = compose_analysis_result(rule_reports, incomplete(), evidence=evidence)
    assert value.verdict == "violation" and len(value.violations) == 1
    assert value.diagnostics["rule_reports"]["R01-R04"]["violations"][0]["rule_id"] == "R03"
    assert value.diagnostics["cash_operation_status"] == "insufficient_evidence"


def test_table_return_without_next_customer_does_not_trigger_r05():
    rule_reports = reports()
    rule_reports["R05"] = run(base() + [table(9, 12)])
    value = compose_analysis_result(rule_reports, cycle().finish(4))
    assert value.verdict == "no_violation" and value.violations == []
    assert rule_reports["R05"].violations == []


def test_cash_to_safe_without_customer_is_no_violation():
    cash = cycle(destination="safe").finish(4)
    assert cash["operations"][0]["associated_session_id"] is None
    assert "MONEY_TO_SAFE" in [e["type"] for e in cash["operations"][0]["events"]]
    value = compose_analysis_result(reports(), cash)
    assert value.verdict == "no_violation" and value.violations == []


def test_confirmed_r05_cannot_be_hidden_by_incomplete_cash_operation():
    rule_reports = reports()
    rule_reports["R05"] = run(base() + [event("CUSTOMER_EXIT", 14), boundary(), table()])
    value = compose_analysis_result(rule_reports, incomplete(), evidence=evidence)
    assert value.verdict == "violation" and len(value.violations) == 1
    assert value.violations[0].type == "CASH_NOT_SECURED_BEFORE_NEXT_CUSTOMER"
    assert value.violations[0].timecode_sec == 16.5
    assert value.diagnostics["rule_reports"]["R05"]["verdict"] == "violation"


def test_uncertain_rule_event_is_not_promoted_or_counted_as_compliance():
    observations = base() + [boundary(), table()]
    observations[-1].update(status="insufficient_evidence", uncertainty=["cash origin unclear"])
    rule_reports = reports()
    rule_reports["R05"] = run(observations)
    value = compose_analysis_result(rule_reports, incomplete(), semantic_events=observations)
    assert value.verdict == "no_violation" and value.violations == []
    assert value.diagnostics["rule_reports"]["R05"]["verdict"] == "insufficient_evidence"
    assert value.diagnostics["semantic_events"][-1]["uncertainty"] == ["cash origin unclear"]


def test_missing_evidence_never_silently_discards_confirmed_alarm():
    rule_reports = reports()
    rule_reports["R05"] = run(base() + [boundary(), table()])
    with pytest.raises(ValueError, match="real evidence"):
        compose_analysis_result(rule_reports, incomplete())


def test_old_cash_report_is_read_as_recognition_not_rule_verdict():
    old = {"verdict": "insufficient_evidence", "operations": [], "violations": []}
    value = compose_analysis_result(reports(), old)
    assert value.verdict == "no_violation"
    assert value.diagnostics["cash_operation_status"] == "insufficient_evidence"
    assert old["verdict"] == "insufficient_evidence"


@pytest.mark.parametrize("alarm", [False, True])
def test_api_roundtrip_keeps_binary_verdict_and_incomplete_diagnostics(tmp_path, alarm):
    settings = Settings(_env_file=None, api_key=KEY, storage_dir=tmp_path)
    rule_reports = reports()
    if alarm:
        rule_reports["R05"] = run(base() + [boundary(), table()])
    value = compose_analysis_result(rule_reports, incomplete(), evidence=evidence)
    with TestClient(create_app(settings)) as client:
        job_id = upload(client)
        jobs = client.app.state.jobs
        assert jobs.complete(jobs.claim_next(30), value)
    # Persist/reload through the actual API contract, not just a function return.
    with TestClient(create_app(settings)) as client:
        body = read_job(client, job_id)
        assert body["verdict"] == ("violation" if alarm else "no_violation")
        assert bool(body["violations"]) == alarm
        assert body["diagnostics"] == value.diagnostics
        assert body["diagnostics"]["cash_operation_status"] == "insufficient_evidence"


def test_existing_analysis_result_contract_without_diagnostics_is_unchanged():
    assert AnalysisResult(verdict="no_violation", violations=[]).model_dump() == {
        "verdict": "no_violation",
        "violations": [],
    }
