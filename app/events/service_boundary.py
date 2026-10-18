"""Camera-calibrated, positive doorway crossings, separate from tracker identity.

Polygons apply to an upper-body anchor, not a floor projection. In this camera
feet are hidden by the desk and bbox centres overlap the glass partition.
Absence alone never retires a customer. No appearance/face identification.
"""

from dataclasses import dataclass, field

from app.events.tracking import center, distance
from app.services.person_roles import point_in_polygon


def person_key(person):
    return person.customer_id or person.track_id


@dataclass
class CustomerVisit:
    customer_id: str
    bbox: tuple
    last_seen: float
    raw_ids: list = field(default_factory=list)
    entered: bool = False
    retired: bool = False
    portal: list = field(default_factory=list)
    exterior: list = field(default_factory=list)


class ServiceBoundaryTracker:
    def __init__(self, config):
        self.config = config
        self.visits = {}
        self.transitions = []
        self.track_changes = []
        self.uncertainties = []

    def region(self, box):
        cfg = self.config
        point = ((box[0] + box[2]) / 2, box[1] + cfg.anchor_height * (box[3] - box[1]))
        return next(
            (
                k
                for k in ("interior", "portal", "exterior")
                if point_in_polygon(point, cfg.zones[k])
            ),
            "unknown",
        )

    def update(self, frame):
        cfg, t = self.config, frame.timestamp_sec
        people, exits, used = [], [], set()
        candidates = [
            p
            for p in frame.people
            if p.role == "customer"
            and p.identity_reliable
            and p.confidence >= cfg.minimum_confidence
        ]
        # Reserve identities of other simultaneously visible bodies before fragment recovery.
        raw_visible = {p.track_id for p in candidates}
        for p in frame.people:
            if p.role != "customer":
                people.append(p)
                continue
            region = self.region(p.bbox)
            choices = []
            if p in candidates:
                for key, visit in self.visits.items():
                    gap = t - visit.last_seen
                    if visit.retired or key in used or not 0 < gap <= cfg.fragment_gap_sec:
                        continue
                    if p.track_id != visit.raw_ids[-1] and visit.raw_ids[-1] in raw_visible:
                        continue
                    d = distance(center(p.bbox), center(visit.bbox))
                    if d <= cfg.fragment_distance:
                        choices.append((d, key))
            choices.sort()
            ambiguous = len(choices) > 1 and choices[1][0] - choices[0][0] < cfg.ambiguity_margin
            visit = self.visits[choices[0][1]] if choices and not ambiguous else None
            if p not in candidates or ambiguous:
                people.append(
                    p.model_copy(
                        update={
                            "customer_id": None,
                            "service_region": region,
                            "identity_reliable": False,
                        }
                    )
                )
                continue
            if visit is None:
                # An unexplained ID jump in the room is not a new customer.
                missing_active = [
                    v
                    for v in self.visits.values()
                    if v.entered
                    and not v.retired
                    and (v.raw_ids[-1] not in raw_visible or v.raw_ids[-1] == p.track_id)
                ]
                if region == "interior" and missing_active:
                    self.uncertainties.append(
                        {
                            "timestamp_sec": t,
                            "raw_track_id": p.track_id,
                            "reason": "unresolved_interior_track_fragment",
                        }
                    )
                    people.append(
                        p.model_copy(
                            update={
                                "customer_id": None,
                                "service_region": region,
                                "identity_reliable": False,
                            }
                        )
                    )
                    continue
                key = f"customer-visit-{len(self.visits) + 1}"
                visit = CustomerVisit(key, p.bbox, t)
                self.visits[key] = visit
            used.add(visit.customer_id)
            if not visit.raw_ids or visit.raw_ids[-1] != p.track_id:
                self.track_changes.append(
                    {
                        "timestamp_sec": t,
                        "customer_id": visit.customer_id,
                        "from_track_id": visit.raw_ids[-1] if visit.raw_ids else None,
                        "to_track_id": p.track_id,
                    }
                )
                visit.raw_ids.append(p.track_id)
            visit.bbox, visit.last_seen = p.bbox, t
            sample = {
                "timestamp_sec": t,
                "confidence": p.confidence,
                "raw_track_id": p.track_id,
                "bbox": list(p.bbox),
                "region": region,
            }
            if region == "interior":
                visit.entered = True
                visit.portal, visit.exterior = [], []
            elif region == "portal":
                if visit.portal and t - visit.portal[-1]["timestamp_sec"] > cfg.fragment_gap_sec:
                    visit.portal = []
                visit.portal.append(sample)
                visit.exterior = []
            elif region == "exterior" and visit.entered and visit.portal:
                if t - visit.portal[-1]["timestamp_sec"] <= cfg.crossing_max_sec:
                    if (
                        visit.exterior
                        and t - visit.exterior[-1]["timestamp_sec"] > cfg.sample_gap_sec
                    ):
                        visit.exterior = []
                    visit.exterior.append(sample)
                    if (
                        len(visit.portal) >= cfg.minimum_frames
                        and len(visit.exterior) >= cfg.minimum_frames
                        and t - visit.exterior[0]["timestamp_sec"] >= cfg.confirm_sec - 1e-8
                    ):
                        visit.retired = True
                        transition = {
                            "customer_id": visit.customer_id,
                            "timestamp_sec": t,
                            "basis": "observed_interior_portal_exterior_crossing",
                            "samples": visit.portal[-cfg.minimum_frames :] + visit.exterior,
                        }
                        self.transitions.append(transition)
                        exits.append(transition)
                else:
                    visit.portal, visit.exterior = [], []
            else:
                visit.exterior = []
            people.append(
                p.model_copy(
                    update={
                        "customer_id": visit.customer_id
                        if visit.entered and not visit.retired
                        else None,
                        "service_region": region,
                    }
                )
            )
        return frame.model_copy(update={"people": people}), exits
