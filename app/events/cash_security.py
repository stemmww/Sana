"""Additional safe/table/ID-return semantics, isolated from the legacy detector.

The producer supplies observable facts, not inferred absence. Pixel motion alone
cannot satisfy the positive evidence contract for safe/lock actions.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.rules.cash_security_schemas import SecurityEvent
from app.services.person_roles import point_in_polygon

NEW_EVENTS = {
    "SAFE_OPENED",
    "MONEY_TO_SAFE",
    "SAFE_CLOSED",
    "SAFE_LOCKED_VISUAL",
    "MONEY_LEFT_ON_TABLE",
    "ID_RETURNED_TO_CUSTOMER",
}
REQUIRED_FACTS = {
    "SAFE_OPENED": {"door_open_visible", "door_transition_visible"},
    "SAFE_CLOSED": {"door_closed_visible", "door_transition_visible"},
    "SAFE_LOCKED_VISUAL": {"key_lock_interaction_visible", "door_closed_visible"},
    "MONEY_TO_SAFE": {"cash_visible", "cash_crosses_safe_opening", "cash_inside_safe_visible"},
    "MONEY_LEFT_ON_TABLE": {"cash_visible", "stationary_on_table", "hand_clear"},
    "ID_RETURNED_TO_CUSTOMER": {
        "id_visible",
        "cashier_to_customer_transition",
        "same_id_confirmed",
    },
}


class SecurityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: SecurityEvent
    facts: set[str] = Field(default_factory=set)
    object_centers: list[tuple[float, float]] = Field(default_factory=list)
    association_reliable: bool = False
    producer_validated: bool = False
    producer: Literal["local_visual", "manual_annotation", "synthetic_example"]


class CashSecurityEventDetector:
    def __init__(self, config, prior_events=()):
        self.config = config
        self.events = list(prior_events)

    def observe(self, observation: SecurityObservation | dict) -> SecurityEvent:
        obs = SecurityObservation.model_validate(observation)
        event = obs.event.model_copy(deep=True)
        if event.type not in NEW_EVENTS:
            raise ValueError("This detector only produces the six additional semantics")
        reasons = list(event.uncertainty)
        missing = REQUIRED_FACTS[event.type] - obs.facts
        if missing:
            reasons.append("missing_positive_facts:" + ",".join(sorted(missing)))
        if not obs.producer_validated:
            reasons.append("visual_producer_unvalidated")
        if not obs.association_reliable:
            reasons.append("object_or_operation_association_unconfirmed")
        if (
            event.sample_count < self.config["minimum_samples"]
            or event.end_sec - event.start_sec < self.config["minimum_duration_sec"]
        ):
            reasons.append("insufficient_temporal_support")
        if len(event.evidence_times) < self.config["minimum_samples"] or any(
            b - a > self.config["maximum_gap_sec"]
            for a, b in zip(event.evidence_times, event.evidence_times[1:], strict=False)
        ):
            reasons.append("missing_or_sparse_positive_samples")
        if not event.evidence_frames:
            reasons.append("evidence_frames_missing")
        if event.confidence < self.config["minimum_confidence"]:
            reasons.append("low_confidence")
        past = [
            e
            for e in self.events
            if e.status == "confirmed"
            and e.confidence >= self.config["minimum_confidence"]
            and e.end_sec <= event.start_sec
            and e.operation_id == event.operation_id
            and e.customer_id == event.customer_id
        ]
        if event.type == "MONEY_LEFT_ON_TABLE":
            if len(obs.object_centers) < self.config["minimum_samples"] or not all(
                point_in_polygon(x, self.config["zones"]["table"])
                and not point_in_polygon(x, self.config["zones"]["safe"])
                for x in obs.object_centers
            ):
                reasons.append("cash_on_table_outside_safe_not_localized")
        if event.type in {"MONEY_TO_SAFE", "SAFE_CLOSED", "SAFE_LOCKED_VISUAL"}:
            states = [
                e
                for e in past
                if e.safe_id == event.safe_id and e.type in {"SAFE_OPENED", "SAFE_CLOSED"}
            ]
            expected = "SAFE_CLOSED" if event.type == "SAFE_LOCKED_VISUAL" else "SAFE_OPENED"
            if not states or states[-1].type != expected:
                reasons.append("preceding_safe_state_unconfirmed")
        if event.type == "ID_RETURNED_TO_CUSTOMER":
            if not any(
                e.type == "ID_TRANSFER_TO_CASHIER" and e.object_id == event.object_id for e in past
            ):
                reasons.append("original_id_transfer_or_identity_unconfirmed")
        if event.type.startswith(("MONEY_", "ID_")) and not event.object_id:
            reasons.append("object_identity_missing")
        if event.type.startswith("MONEY_") and not event.money_group_id:
            reasons.append("money_origin_identity_missing")
        if not event.operation_id or not event.customer_id:
            reasons.append("origin_operation_missing")
        if (event.type.startswith("SAFE_") or event.type == "MONEY_TO_SAFE") and not event.safe_id:
            reasons.append("safe_identity_missing")
        # Never upgrade a producer's explicit uncertainty automatically.
        if event.status != "confirmed":
            reasons.append("producer_did_not_confirm_observation")
        event.status = "insufficient_evidence" if reasons else "confirmed"
        event.uncertainty = list(dict.fromkeys(reasons))
        event.observation_basis = (
            "Visible key/lock action only; physical locking is not established"
            if event.type == "SAFE_LOCKED_VISUAL"
            else ";".join(sorted(obs.facts))
        )
        self.events.append(event)
        return event
