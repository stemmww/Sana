"""Real-video evidence adapter for the common external result composer."""

import base64
from pathlib import Path

from app.rules.cash_security import evaluate_cash_security
from app.rules.cash_security_schemas import CashSecurityRequest, SecurityEvent
from app.rules.schemas import RuleReport
from app.schemas import EvidenceFrame
from app.services.analysis_result import compose_analysis_result


def security_request_from_legacy(legacy):
    """Adapt existing semantic events, retaining their service origin and confidence.

    No events are invented from a CashOperation or a missing cash/safe observation.
    Service start/end are the existing Rule Engine's derived events.
    """
    events = []
    for raw in legacy["rule_request"]["events"]:
        session = next(
            (
                s
                for s in legacy["rule_result"]["sessions"]
                if s["customer_id"] == raw.get("customer_id")
                and s["start_sec"] <= raw["end_sec"]
                and (s["end_sec"] is None or raw["end_sec"] <= s["end_sec"])
            ),
            None,
        )
        value = {**raw, "operation_id": session["session_id"] if session else None}
        if raw["type"].startswith(("MONEY_", "ID_", "SAFE_")) and session is None:
            value.update(status="insufficient_evidence", uncertainty=["origin_session_unconfirmed"])
        events.append(SecurityEvent.model_validate(value))
    lookup = {e.event_id: e for e in events}
    for derived in legacy["rule_result"]["derived_events"]:
        if derived["type"] not in {"SERVICE_STARTED", "SERVICE_ENDED"}:
            continue
        trigger = lookup[derived["trigger_event_id"]]
        events.append(
            SecurityEvent(
                event_id=f"derived-{derived['type']}-{trigger.event_id}",
                type=derived["type"],
                start_sec=trigger.start_sec,
                end_sec=trigger.end_sec,
                source=trigger.source,
                confidence=trigger.confidence,
                sample_count=trigger.sample_count,
                operation_id=derived["session_id"],
                customer_id=derived["customer_id"],
            )
        )
    return CashSecurityRequest(
        duration_sec=legacy["rule_request"]["duration_sec"],
        events=sorted(events, key=lambda e: e.end_sec),
    )


def video_evidence(video: Path, alarm):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or count <= 0:
            raise ValueError("Cannot decode violation evidence video")
        t = alarm["timecode_sec"]
        if not 0 <= t <= count / fps:
            raise ValueError("Violation timestamp outside video")
        frame_index = min(int(round(t * fps)), int(count) - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Cannot read confirmed violation evidence")
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Cannot encode confirmed violation evidence")
        return [
            EvidenceFrame(
                timestamp_sec=frame_index / fps,
                image_base64=base64.b64encode(encoded).decode("ascii"),
            )
        ]
    finally:
        capture.release()


def assemble_video_result(video, legacy, cash_operation=None, security_request=None):
    request = (
        security_request if security_request is not None else security_request_from_legacy(legacy)
    )
    security = evaluate_cash_security(request)
    return compose_analysis_result(
        {"R01-R04": RuleReport.model_validate(legacy["rule_result"]), "R05": security},
        cash_operation,
        semantic_events=list(legacy["events"])
        + [{**e.model_dump(mode="json"), "event_stream": "cash_security"} for e in request.events]
        + [
            {**event, "event_stream": "cash_operation"}
            for op in (cash_operation or {}).get("operations", [])
            for event in op["events"]
        ],
        evidence=lambda alarm: video_evidence(video, alarm),
    )
