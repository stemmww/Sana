"""Customer-optional cash operations. This ledger never supplies events to R01–R05.

A stable visible table return and a positively observed safe deposit are alternative
endpoints. Missing detection, an absent customer, or missing writing is not a violation.
"""

from dataclasses import asdict, dataclass, field
from math import hypot, isfinite
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CashObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: float = Field(ge=0, allow_inf_nan=False)
    logical_cash_id: str = Field(min_length=1)
    location: Literal["cashier", "table", "counter", "safe", "unknown"]
    center: tuple[float, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    cashier_track_id: str | None = None
    associated_session_id: str | None = None
    ambiguous: bool = False
    evidence_frame: str
    # Safe overlap or disappearance is deliberately insufficient.
    visible_inside_safe: bool = False
    crossed_safe_opening: bool = False
    safe_open_observed: bool = False
    safe_producer_validated: bool = False

    @field_validator("center")
    @classmethod
    def normalized_point(cls, point):
        if any(not isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError("Cash center must be finite and normalized")
        return point


@dataclass
class CashOperation:
    operation_id: str
    start_time: float
    end_time: float | None
    associated_session_id: str | None
    cashier_track_id: str
    logical_cash_ids: list[str]
    events: list[dict] = field(default_factory=list)
    completion_status: str = "in_progress"
    evidence_confidence: float = 0
    uncertainty: list[str] = field(default_factory=list)


@dataclass
class TrackState:
    candidate: str | None = None
    samples: list[CashObservation] = field(default_factory=list)
    stable: str | None = None
    stable_samples: list[CashObservation] = field(default_factory=list)
    last_seen: float = -1
    operation: CashOperation | None = None
    continuity_broken: bool = False


class CashOperationDetector:
    """Only same-logical-object positive transitions; never merge neighboring stacks."""

    minimum_samples = 3
    minimum_span = 0.4
    max_gap = 0.8
    minimum_motion = 0.012
    minimum_confidence = 0.75

    def __init__(self):
        self.states: dict[str, TrackState] = {}
        self.operations: list[CashOperation] = []
        self.timestamp = -1
        self.uncertainties: list[dict] = []

    def update(self, timestamp, observations, *, profile_valid=True):
        if not isfinite(timestamp) or timestamp <= self.timestamp:
            raise ValueError("Operation timestamps must increase")
        self.timestamp = timestamp
        if not profile_valid:
            self.uncertainties.append({"timestamp": timestamp, "reason": "camera_unverified"})
            for state in self.states.values():
                self.break_continuity(state, "camera_unverified")
            return
        parsed = [CashObservation.model_validate(o) for o in observations]
        keys = [o.logical_cash_id for o in parsed]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate logical cash observation")
        for obs in parsed:
            if abs(obs.timestamp - timestamp) > 1e-6:
                raise ValueError("Cash observation timestamp differs from frame")
            state = self.states.setdefault(obs.logical_cash_id, TrackState())
            if obs.ambiguous or obs.confidence < 0.25:
                self.break_continuity(state, "ambiguous_cash_identity")
                continue
            if state.last_seen >= 0 and timestamp - state.last_seen > self.max_gap:
                self.break_continuity(state, "cash_observation_gap")
            state.last_seen = timestamp
            location = obs.location
            if location == "safe" and not all(
                [
                    obs.visible_inside_safe,
                    obs.crossed_safe_opening,
                    obs.safe_open_observed,
                    obs.safe_producer_validated,
                ]
            ):
                location = "unknown"
                if state.operation:
                    self.note(state.operation, "safe_deposit_not_visually_confirmed")
            if location == "cashier" and not obs.cashier_track_id:
                location = "unknown"
            if location != state.candidate:
                state.candidate, state.samples = location, []
            state.samples.append(obs)
            # Bounded evidence window; old observations cannot confirm a new action.
            state.samples = [o for o in state.samples if timestamp - o.timestamp <= 2]
            if location == "unknown":
                # No inference across an unobserved intermediate location.
                self.break_continuity(state, "cash_location_unknown")
                continue
            if not self.stable_enough(state.samples):
                continue
            previous, old = state.stable, state.stable_samples
            moved = self.motion(state.samples)
            displacement = (
                hypot(*(a - b for a, b in zip(old[-1].center, obs.center, strict=True)))
                if old
                else 0
            )
            if (
                location == "cashier"
                and state.operation is None
                and (
                    moved >= self.minimum_motion
                    or (previous in {"table", "counter"} and displacement >= self.minimum_motion)
                )
            ):
                self.start(state, obs)
            operation = state.operation
            if operation and operation.cashier_track_id != (
                obs.cashier_track_id or operation.cashier_track_id
            ):
                self.break_continuity(state, "cashier_ownership_changed")
                continue
            if operation and previous != location and old:
                transition = (previous, location)
                kind = {
                    ("table", "cashier"): "MONEY_FROM_TABLE",
                    ("cashier", "counter"): "MONEY_TO_COUNTER",
                    ("counter", "cashier"): "MONEY_FROM_COUNTER",
                    ("cashier", "table"): "MONEY_TO_TABLE",
                    ("cashier", "safe"): "MONEY_TO_SAFE",
                }.get(transition)
                if kind and displacement >= self.minimum_motion:
                    self.emit(operation, kind, old + state.samples)
                    if location in {"table", "safe"}:
                        self.try_complete(operation, obs.timestamp)
                elif kind:
                    self.note(operation, "location_change_without_sufficient_cash_movement")
            if previous != location:
                state.stable, state.stable_samples = location, list(state.samples)
            else:
                state.stable_samples = list(state.samples)
            if operation and operation.completion_status == "completed":
                state.operation = None

    @staticmethod
    def motion(samples):
        return max(
            (
                hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
                for a in samples
                for b in samples
            ),
            default=0,
        )

    def stable_enough(self, samples):
        return len(samples) >= self.minimum_samples and (
            samples[-1].timestamp - samples[0].timestamp >= self.minimum_span - 1e-8
        )

    def start(self, state, obs):
        operation = CashOperation(
            operation_id=f"cash-operation-{len(self.operations) + 1}",
            start_time=state.samples[0].timestamp,
            end_time=None,
            associated_session_id=obs.associated_session_id,
            cashier_track_id=obs.cashier_track_id,
            logical_cash_ids=[obs.logical_cash_id],
        )
        self.operations.append(operation)
        state.operation = operation
        self.emit(operation, "MONEY_HELD_BY_CASHIER", state.samples)
        if state.continuity_broken:
            self.note(operation, "origin_before_start_unconfirmed")

    @staticmethod
    def note(operation, reason):
        if reason not in operation.uncertainty:
            operation.uncertainty.append(reason)

    def break_continuity(self, state, reason):
        state.stable, state.candidate = None, None
        state.samples, state.stable_samples = [], []
        state.continuity_broken = True
        if state.operation:
            self.note(state.operation, reason)

    def emit(self, operation, kind, samples):
        # Evidence score, not a calibrated correctness probability.
        confidence = round(0.45 * mean(o.confidence for o in samples) + 0.55, 4)
        status = "confirmed" if confidence >= self.minimum_confidence else "uncertain"
        times = sorted({o.timestamp for o in samples})
        event = {
            "type": kind,
            "operation_id": operation.operation_id,
            "associated_session_id": operation.associated_session_id,
            "logical_cash_ids": list(operation.logical_cash_ids),
            "timestamp": times[-1],
            "start_time": times[0],
            "end_time": times[-1],
            "confidence": confidence,
            "status": status,
            "observed_features": sorted({o.location for o in samples}),
            "evidence_times": times,
            "evidence_frames": sorted({o.evidence_frame for o in samples}),
            "uncertainty": [] if status == "confirmed" else ["low_observation_confidence"],
        }
        operation.events.append(event)
        operation.evidence_confidence = min(e["confidence"] for e in operation.events)

    def try_complete(self, operation, timestamp):
        confirmed = [e["type"] for e in operation.events if e["status"] == "confirmed"]
        # Ordered counter cycle; table and safe are equally acceptable final destinations.
        step = 0
        for kind in confirmed:
            if step == 0 and kind == "MONEY_TO_COUNTER":
                step = 1
            elif step == 1 and kind == "MONEY_FROM_COUNTER":
                step = 2
            elif step == 2 and kind in {"MONEY_TO_TABLE", "MONEY_TO_SAFE"}:
                step = 3
        if step == 3 and not operation.uncertainty:
            operation.end_time, operation.completion_status = timestamp, "completed"
            operation.events.append(
                {
                    "type": "CASH_OPERATION_COMPLETED",
                    "operation_id": operation.operation_id,
                    "timestamp": timestamp,
                    "status": "confirmed",
                    "confidence": operation.evidence_confidence,
                    "evidence_frames": sorted(
                        {p for e in operation.events for p in e["evidence_frames"]}
                    ),
                    "uncertainty": [],
                }
            )

    def finish(self, duration):
        for operation in self.operations:
            if operation.completion_status != "completed":
                operation.completion_status = "insufficient_evidence"
                self.note(operation, "complete_counter_and_destination_sequence_unconfirmed")
        return {
            "duration": duration,
            "operations": [asdict(o) for o in self.operations],
            "status": "completed"
            if self.operations and all(o.completion_status == "completed" for o in self.operations)
            else "insufficient_evidence",
            "status_scope": "Recognition of cash operations only; no business rules evaluated here",
            "uncertainties": self.uncertainties,
            "notebook_writing": "deferred_optional_observation",
        }
