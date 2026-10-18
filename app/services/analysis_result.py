"""Compose an external verdict without confusing recognition gaps with violations.

Rule reports remain unchanged in diagnostics. no_violation means no confirmed
violation was detected; it never certifies complete procedural compliance.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy

from app.rules.cash_security_schemas import CashSecurityReport
from app.rules.schemas import RuleReport
from app.schemas import AnalysisResult, EvidenceFrame, Violation


def compose_analysis_result(
    rule_reports: Mapping[str, RuleReport | CashSecurityReport],
    cash_operation: dict | None = None,
    *,
    semantic_events: list[dict] | None = None,
    evidence: Callable[[dict], list[EvidenceFrame]] | None = None,
) -> AnalysisResult:
    """Only trusted Rule Engine violation lists decide the external verdict.

    Recognition/event candidates are diagnostics, never violations. Evidence failures
    raise an error rather than dropping a confirmed violation from the result.
    """
    reports = {name: report.model_dump(mode="json") for name, report in rule_reports.items()}
    cash = deepcopy(cash_operation)
    # Read older diagnostic artifacts without treating their old `verdict` as a rule decision.
    cash_status = "not_evaluated"
    if cash is not None:
        cash_status = cash.get("status", cash.get("verdict", "insufficient_evidence"))
        if cash_status == "completed_observed_operations":
            cash_status = "completed"
        cash["status"] = cash_status
        cash.pop("verdict", None)
    reasons = []
    if cash_status == "insufficient_evidence":
        reasons.append({"source": "CashOperation", "code": "cash_recognition_incomplete"})
    if cash:
        if cash.get("reason"):
            reasons.append({"source": "CashOperation", "code": cash["reason"]})
        reasons.extend(
            {"source": "CashOperation", **item} for item in cash.get("uncertainties", [])
        )
        for operation in cash.get("operations", []):
            reasons.extend(
                {
                    "source": "CashOperation",
                    "operation_id": operation["operation_id"],
                    "code": reason,
                }
                for reason in operation.get("uncertainty", [])
            )
    for name, report in reports.items():
        if report["verdict"] == "insufficient_evidence":
            reasons.append({"source": name, "code": "rule_evidence_incomplete"})
        reasons.extend({"source": name, **u} for u in report.get("uncertainties", []))
    for name in {"R01-R04", "R05"} - reports.keys():
        reasons.append({"source": name, "code": "rules_not_evaluated"})
    if cash is None:
        reasons.append({"source": "CashOperation", "code": "cash_operation_not_evaluated"})

    violations, seen = [], set()
    for report in reports.values():
        for alarm in report["violations"]:
            # R01-R05 currently output only confirmed violations. This guard also
            # prevents explicit uncertain records from being promoted by an adapter.
            if alarm.get("status", "confirmed") != "confirmed":
                reasons.append(
                    {
                        "source": "RuleEngine",
                        "code": "unconfirmed_alarm_excluded",
                        "alarm": deepcopy(alarm),
                    }
                )
                continue
            key = (
                alarm.get("rule_id"),
                alarm["type"],
                alarm["timecode_sec"],
                alarm.get("customer_id"),
                alarm.get("operation_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            if evidence is None:
                raise ValueError("Confirmed violations require real evidence frames")
            violations.append(
                Violation(
                    type=alarm["type"],
                    timecode_sec=alarm["timecode_sec"],
                    description=alarm["description"],
                    evidence_frames=evidence(alarm),
                )
            )
    return AnalysisResult(
        verdict="violation" if violations else "no_violation",
        violations=violations,
        diagnostics={
            "analysis_status": "incomplete" if reasons else "complete",
            "cash_operation_status": cash_status,
            "cash_operation": cash,
            "rule_reports": reports,
            "insufficient_evidence_reasons": reasons,
            "semantic_events": deepcopy(semantic_events or []),
            "verdict_meaning": "presence_or_absence_of_detected_confirmed_violations",
            "full_procedural_compliance": "not_established",
        },
    )
