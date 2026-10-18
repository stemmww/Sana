"""R05 needs positive same-money table evidence at a new customer boundary."""

from app.rules.cash_security_schemas import CashSecurityReport, CashSecurityRequest

WORKFLOW = (
    "ID_TRANSFER_TO_CASHIER",
    "ID_CHECK",
    "MONEY_TRANSFER_TO_CASHIER",
    "MONEY_TO_COUNTER",
    "MONEY_FROM_COUNTER",
    "SAFE_OPENED",
    "MONEY_TO_SAFE",
    "SAFE_CLOSED",
    "SAFE_LOCKED_VISUAL",
    "ID_RETURNED_TO_CUSTOMER",
    "CUSTOMER_EXIT",
)


def confirmed(event, request):
    return event.status == "confirmed" and event.confidence >= request.minimum_confidence


def table_at(event, timestamp, max_gap):
    """Positive samples must bracket the boundary; no stale last-known-position inference."""
    return any(
        a <= timestamp <= b and b - a <= max_gap + 1e-9
        for a, b in zip(event.evidence_times, event.evidence_times[1:], strict=False)
    )


def workflow(request):
    good = [e for e in request.events if confirmed(e, request)]
    operations = []
    for start in good:
        if start.type != "SERVICE_STARTED":
            continue
        if any(o["operation_id"] == start.operation_id for o in operations):
            continue
        prior_presence = [
            e
            for e in good
            if e.type in {"CASHIER_PRESENT", "CASHIER_EXIT", "CASHIER_ABSENT"}
            and e.end_sec <= start.start_sec
        ]
        cashier_ready = bool(prior_presence) and prior_presence[-1].type == "CASHIER_PRESENT"
        steps, issues, last, card, group, safe = [], [], start, None, None, None
        if not cashier_ready:
            issues.append("cashier_presence_before_service_unconfirmed")
        for event in good:
            if event.operation_id != start.operation_id or event.customer_id != start.customer_id:
                continue
            if len(steps) == len(WORKFLOW) or event.type != WORKFLOW[len(steps)]:
                continue
            if event.start_sec < last.end_sec:
                issues.append(f"overlapping_steps:{event.event_id}")
                continue
            if event.type.startswith("ID_"):
                if card and event.object_id != card:
                    issues.append(f"id_identity_changed:{event.event_id}")
                    continue
                card = event.object_id
            if event.type.startswith("MONEY_"):
                if group and event.money_group_id != group:
                    issues.append(f"money_identity_changed:{event.event_id}")
                    continue
                group = event.money_group_id
            if event.safe_id:
                if safe and event.safe_id != safe:
                    issues.append(f"safe_identity_changed:{event.event_id}")
                    continue
                safe = event.safe_id
            steps.append({"type": event.type, "event_id": event.event_id, "end_sec": event.end_sec})
            last = event
        complete = len(steps) == len(WORKFLOW) and cashier_ready and not issues
        operations.append(
            {
                "operation_id": start.operation_id,
                "customer_id": start.customer_id,
                "status": "completed" if complete else "insufficient_evidence",
                "state": "COMPLETED"
                if complete
                else WORKFLOW[len(steps)]
                if len(steps) < len(WORKFLOW)
                else "CASHIER_PRESENT",
                "accepted_steps": steps,
                "missing_steps": list(WORKFLOW[len(steps) :]),
                "uncertainty": issues,
                "lock_semantics": "Visible key/lock action only, not proof of physical locking",
            }
        )
    return operations


def evaluate_cash_security(request: CashSecurityRequest) -> CashSecurityReport:
    good = [e for e in request.events if confirmed(e, request)]
    ignored = [
        {"code": "unconfirmed_event", "event_id": e.event_id, "reasons": e.uncertainty}
        for e in request.events
        if not confirmed(e, request)
    ]
    withdrawals = [e for e in good if e.type == "MONEY_FROM_COUNTER"]
    boundaries = [e for e in good if e.type in {"CUSTOMER_ENTER", "SERVICE_STARTED"}]
    checks, violations, alarms = [], [], set()
    for removed in withdrawals:
        key = (removed.operation_id, removed.money_group_id)
        later = [
            b
            for b in boundaries
            if b.start_sec >= removed.end_sec
            and (
                b.customer_id != removed.customer_id
                or b.type == "SERVICE_STARTED"
                and b.operation_id != removed.operation_id
            )
        ]
        # The first confirmed next-customer boundary is decisive, not subsequent services.
        if not later:
            checks.append(
                {
                    "operation_id": key[0],
                    "money_group_id": key[1],
                    "status": "pending",
                    "reason": "no_observed_next_customer_boundary",
                }
            )
            continue
        boundary = min(later, key=lambda b: b.end_sec)
        t = boundary.end_sec
        table = [
            e
            for e in good
            if e.type == "MONEY_LEFT_ON_TABLE"
            and (e.operation_id, e.money_group_id) == key
            and e.customer_id == removed.customer_id
            and e.start_sec >= removed.end_sec
            and table_at(e, t, request.max_table_observation_gap_sec)
        ]
        secured = [
            e
            for e in good
            if e.type == "MONEY_TO_SAFE"
            and (e.operation_id, e.money_group_id) == key
            and e.customer_id == removed.customer_id
            and e.start_sec >= removed.end_sec
            and e.end_sec <= t
        ]
        record = {
            "operation_id": key[0],
            "money_group_id": key[1],
            "customer_id": removed.customer_id,
            "next_customer_id": boundary.customer_id,
            "boundary_event_id": boundary.event_id,
            "timecode_sec": t,
        }
        if table and secured:
            record.update(
                status="insufficient_evidence", reason="contradictory_safe_and_table_identity"
            )
        elif table:
            proof = min(table, key=lambda e: e.end_sec)
            record.update(
                status="violation", reason="same_money_positively_observed_on_table_at_boundary"
            )
            if key not in alarms:
                alarms.add(key)
                violations.append(
                    {
                        "rule_id": "R05",
                        "type": "CASH_NOT_SECURED_BEFORE_NEXT_CUSTOMER",
                        "timecode_sec": t,
                        "detected_at_sec": max(t, proof.end_sec),
                        "operation_id": key[0],
                        "money_group_id": key[1],
                        "customer_id": removed.customer_id,
                        "next_customer_id": boundary.customer_id,
                        "evidence_event_ids": [removed.event_id, proof.event_id, boundary.event_id],
                        "description": (
                            "Деньги предыдущей операции положительно наблюдаются на столе "
                            "при входе следующего клиента или начале новой сессии."
                        ),
                    }
                )
        elif secured:
            record.update(
                status="no_violation",
                reason="same_money_positively_observed_entering_safe",
                evidence_event_ids=[removed.event_id, secured[-1].event_id, boundary.event_id],
            )
        else:
            record.update(
                status="insufficient_evidence", reason="no_fresh_same_money_table_or_safe_evidence"
            )
        checks.append(record)
    if not withdrawals:
        ignored.append({"code": "counter_withdrawal_unconfirmed"})
    for c in checks:
        if c["status"] in {"pending", "insufficient_evidence"}:
            ignored.append({"code": c["reason"], "operation_id": c["operation_id"]})
    verdict = (
        "violation"
        if violations
        else "no_violation"
        if checks
        and all(c["status"] == "no_violation" for c in checks)
        and not any(
            e.type == "MONEY_FROM_COUNTER" and not confirmed(e, request) for e in request.events
        )
        else "insufficient_evidence"
    )
    return CashSecurityReport(
        verdict=verdict,
        violations=violations,
        boundary_checks=checks,
        operations=workflow(request),
        uncertainties=ignored,
    )
