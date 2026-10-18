"""Translate existing cash tracks and pose wrists into conservative location evidence."""

from math import hypot

from app.events.cash_operations import CashObservation
from app.services.person_roles import point_in_polygon
from app.services.role_tracking import box_iou


def cash_observations(frame, pose_frame, zones, width, height, evidence_frame):
    wrists = []
    people = [p for p in frame["people"] if p["identity_reliable"]]
    if pose_frame and 0 <= frame["timestamp_sec"] - pose_frame["timestamp_sec"] <= 0.15:
        for pose in pose_frame["poses"]:
            matches = []
            for person in people:
                box = [v * (width if i % 2 == 0 else height) for i, v in enumerate(person["bbox"])]
                overlap = box_iou(box, pose["bbox_px"])
                if overlap >= 0.3:
                    matches.append((overlap, person))
            matches.sort(key=lambda pair: pair[0], reverse=True)
            if not matches or len(matches) > 1 and matches[0][0] - matches[1][0] < 0.15:
                continue
            owner = matches[0][1]
            for joint in ("9", "10"):
                x, y, confidence = pose["joints"][joint]
                if confidence >= 0.35 and x > 0 and y > 0:
                    wrists.append({"xy": (x, y), "owner": owner})
    observations = []
    for obj in frame["objects"]:
        if obj["label"] != "cash":
            continue
        point = obj["center"]
        pixel = (point[0] * width, point[1] * height)
        nearby = [w for w in wrists if hypot(pixel[0] - w["xy"][0], pixel[1] - w["xy"][1]) <= 22]
        owners = {w["owner"]["track_id"] for w in nearby}
        cashier = next(
            (
                w["owner"]["track_id"]
                for w in nearby
                if w["owner"]["role"] == "cashier" and len(owners) == 1
            ),
            None,
        )
        location = "unknown"
        # Two visible wrists from the cashier are required for hand-clear table/counter evidence.
        cashiers = {w["owner"]["track_id"] for w in wrists if w["owner"]["role"] == "cashier"}
        clear = (
            len(cashiers) == 1
            and len([w for w in wrists if w["owner"]["role"] == "cashier"]) >= 2
            and not nearby
        )
        in_safe = point_in_polygon(point, zones["safe"])
        in_counter = point_in_polygon(point, zones["counter"])
        if in_safe:
            location = "safe"  # Core rejects this without explicit visible-interior proof.
        elif in_counter and clear:
            location = "counter"
        elif cashier:
            location = "cashier"
        elif clear and point_in_polygon(point, zones["table"]):
            location = "table"
        observations.append(
            CashObservation(
                timestamp=frame["timestamp_sec"],
                logical_cash_id=obj["logical_object_id"],
                location=location,
                center=point,
                confidence=obj["confidence"],
                cashier_track_id=cashier,
                associated_session_id=frame["active_session"],
                ambiguous=obj.get("association_ambiguous", False),
                evidence_frame=evidence_frame,
            )
        )
    return observations
