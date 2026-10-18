"""Opt-in cash-security contract. Legacy R01-R04 schemas remain unchanged."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rules.schemas import EventType, Span

SecurityEventType = (
    EventType
    | Literal[
        "SERVICE_STARTED",
        "SERVICE_ENDED",
        "SAFE_OPENED",
        "MONEY_TO_SAFE",
        "SAFE_CLOSED",
        "SAFE_LOCKED_VISUAL",
        "MONEY_LEFT_ON_TABLE",
        "ID_RETURNED_TO_CUSTOMER",
    ]
)
EvidenceStatus = Literal["confirmed", "uncertain", "insufficient_evidence"]


class SecurityEvent(Span):
    event_id: str = Field(min_length=1)
    type: SecurityEventType
    status: EvidenceStatus = "confirmed"
    source: Literal["local_tracker", "local_visual", "manual_annotation", "synthetic_example"]
    confidence: float = Field(default=1, ge=0, le=1, allow_inf_nan=False)
    sample_count: int = Field(default=2, ge=2)
    # Origin operation persists after its customer leaves. Never reassign old money to new service.
    operation_id: str | None = None
    customer_id: str | None = None
    person_id: str | None = None
    object_id: str | None = None
    money_group_id: str | None = None
    safe_id: str | None = None
    evidence_times: list[float] = Field(default_factory=list)
    evidence_frames: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    observation_basis: str | None = None

    @model_validator(mode="after")
    def evidence_contract(self):
        if self.evidence_times != sorted(set(self.evidence_times)):
            raise ValueError("evidence_times must be unique and increasing")
        if any(not self.start_sec <= t <= self.end_sec for t in self.evidence_times):
            raise ValueError("Evidence time outside event interval")
        if len(self.evidence_times) > self.sample_count:
            raise ValueError("More evidence times than observed samples")
        if self.status != "confirmed":
            if not self.uncertainty:
                raise ValueError("Unconfirmed observations require an explanation")
            return self
        if self.uncertainty:
            raise ValueError("Unresolved uncertainty requires an unconfirmed status")
        if self.type.startswith(("MONEY_", "ID_", "SAFE_")):
            if not self.customer_id or not self.operation_id:
                raise ValueError("Confirmed actions require their origin operation and customer")
        if self.type.startswith(("MONEY_", "ID_")) and not self.object_id:
            raise ValueError("Confirmed object actions require object identity")
        if self.type.startswith("MONEY_") and not self.money_group_id:
            raise ValueError("Confirmed money actions require positive common-group identity")
        if self.type.startswith("SAFE_") or self.type == "MONEY_TO_SAFE":
            if not self.safe_id:
                raise ValueError("Confirmed safe actions require safe_id")
        if (
            self.type
            in {
                "CUSTOMER_ENTER",
                "CUSTOMER_EXIT",
                "CUSTOMER_PRESENT",
                "SERVICE_STARTED",
                "SERVICE_ENDED",
            }
            and not self.customer_id
        ):
            raise ValueError("Customer boundary requires customer_id")
        if self.type == "SERVICE_STARTED" and not self.operation_id:
            raise ValueError("Service start requires operation_id")
        if self.type == "MONEY_LEFT_ON_TABLE" and len(self.evidence_times) < 2:
            raise ValueError("Table persistence requires actual positive sample timestamps")
        return self


class CashSecurityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration_sec: float = Field(gt=0, allow_inf_nan=False)
    events: list[SecurityEvent] = Field(default_factory=list, max_length=20000)
    minimum_confidence: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    max_table_observation_gap_sec: float = Field(default=0.8, gt=0, le=2, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_timeline(self):
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("Duplicate event IDs")
        if any(e.end_sec > self.duration_sec for e in self.events):
            raise ValueError("Event outside video")
        if any(b.end_sec < a.end_sec for a, b in zip(self.events, self.events[1:], strict=False)):
            raise ValueError("Events must be sorted by end_sec")
        return self


class CashSecurityReport(BaseModel):
    scope: str = "R05 and cash-to-safe workflow only; R01-R04 unchanged"
    verdict: Literal["violation", "no_violation", "insufficient_evidence"]
    violations: list[dict]
    boundary_checks: list[dict]
    operations: list[dict]
    uncertainties: list[dict]
