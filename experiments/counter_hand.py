"""Pure wrist observation/sequence diagnostic. No production event imports or I/O."""

from __future__ import annotations

import math
from collections import Counter

import cv2
import numpy as np


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def distance(point, polygon):
    return max(0.0, -cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(point), True))


def observations(pose_frames, role_frames, p):
    """Causal geometric association with cached roles and independent wrist tracks."""
    result, tracks, ri = [], [], -1
    for frame in pose_frames:
        t, w, h = frame["timestamp_sec"], frame["width"], frame["height"]
        scale = max(x[0] for x in p["counter_zone"]) * w - min(x[0] for x in p["counter_zone"]) * w
        polygon = [[x * w, y * h] for x, y in p["counter_output_zone"]]
        core = [[x * w, y * h] for x, y in p["output_core"]]
        while ri + 1 < len(role_frames) and role_frames[ri + 1]["timestamp_sec"] <= t + 1e-6:
            ri += 1
        people = (
            role_frames[ri]["people"]
            if ri >= 0 and t - role_frames[ri]["timestamp_sec"] <= p["role_cache_max_age_sec"]
            else []
        )
        hands, used = [], set()
        for pose in frame["poses"]:
            joints = pose["joints"]
            torso = [
                joints[str(k)] for k in [5, 6, 11, 12] if joints[str(k)][2] >= p["joint_confidence"]
            ]
            anchor = np.mean([j[:2] for j in torso], axis=0).tolist() if torso else None
            matches = []
            for person in people:
                if (
                    not person.get("identity_reliable", False)
                    or person["role"] == "unknown"
                    or anchor is None
                ):
                    continue
                box = [v * (w if n % 2 == 0 else h) for n, v in enumerate(person["bbox"])]
                overlap = iou(pose["bbox_px"], box)
                if (
                    overlap >= p["pose_role_min_iou"]
                    and box[0] <= anchor[0] <= box[2]
                    and box[1] <= anchor[1] <= box[3]
                ):
                    matches.append((overlap, person))
            matches.sort(key=lambda m: m[0], reverse=True)
            unique = bool(matches) and (
                len(matches) == 1 or matches[0][0] - matches[1][0] >= p["pose_role_margin"]
            )
            person = matches[0][1] if unique else None
            for side, shoulder, elbow, wrist in [("left", 5, 7, 9), ("right", 6, 8, 10)]:
                sj, ej, wj = [joints[str(k)] for k in [shoulder, elbow, wrist]]
                if wj[2] < p["wrist_confidence"] or min(wj[:2]) <= 0:
                    continue
                arm_reliable = min(sj[2], ej[2]) >= p["joint_confidence"]
                role = person["role"] if person and arm_reliable else "unknown"
                body_id = person["track_id"] if person else None
                hand = {
                    "timestamp_sec": t,
                    "frame_index": frame["frame_index"],
                    "side": side,
                    "xy": wj[:2],
                    "wrist_confidence": wj[2],
                    "arm_confidence": min(sj[2], ej[2], wj[2]),
                    "shoulder": sj,
                    "elbow": ej,
                    "torso_anchor": anchor,
                    "pose_bbox_px": pose["bbox_px"],
                    "body_track_id": body_id,
                    "instant_owner": role,
                    "owner": "unknown",
                    "owner_confidence": 0.0,
                    "owner_iou": matches[0][0] if unique else None,
                    "distance_widths": distance(wj[:2], polygon) / scale,
                    "core_distance_widths": distance(wj[:2], core) / scale,
                    "inside_output_zone": distance(wj[:2], polygon) == 0,
                    "uncertainty": [],
                }
                if not unique:
                    hand["uncertainty"].append("No unique torso-to-cached-role geometric match.")
                if not arm_reliable:
                    hand["uncertainty"].append(
                        "Shoulder/elbow confidence insufficient to establish acting arm."
                    )
                choices = []
                for idx, chain in enumerate(tracks):
                    last = chain[-1]
                    if (
                        idx in used
                        or side != last["side"]
                        or t - last["timestamp_sec"] > p["hand_track_max_gap_sec"]
                    ):
                        continue
                    if body_id and last["body_track_id"] and body_id != last["body_track_id"]:
                        continue
                    step = math.dist(hand["xy"], last["xy"]) / scale
                    torso_step = (
                        math.dist(anchor, last["torso_anchor"]) / scale
                        if anchor and last["torso_anchor"]
                        else float("inf")
                    )
                    if step <= p["hand_track_max_step_widths"] and torso_step <= 0.8:
                        choices.append((step, idx))
                choices.sort()
                ambiguous = (
                    len(choices) > 1
                    and choices[1][0] - choices[0][0] < p["hand_track_ambiguity_widths"]
                )
                if choices and not ambiguous:
                    idx = choices[0][1]
                else:
                    idx = len(tracks)
                    tracks.append([])
                    if ambiguous:
                        hand["uncertainty"].append("Competing wrist tracks; started a new track.")
                hand["track_id"] = f"hand-{idx + 1}"
                tracks[idx].append(hand)
                used.add(idx)
                history = tracks[idx][-p["owner_min_samples"] :]
                if (
                    len(history) >= p["owner_min_samples"]
                    and role != "unknown"
                    and all(
                        n["instant_owner"] == role and n["body_track_id"] == body_id
                        for n in history
                    )
                    and t - history[0]["timestamp_sec"] >= p["owner_min_span_sec"]
                    and not ambiguous
                ):
                    hand["owner"] = role
                    hand["owner_confidence"] = min(n["arm_confidence"] for n in history)
                else:
                    hand["uncertainty"].append(
                        "Insufficient consistent temporal ownership evidence."
                    )
                hands.append(hand)
        result.append(
            {
                "timestamp_sec": t,
                "frame_index": frame["frame_index"],
                "width": w,
                "height": h,
                "counter_width_px": scale,
                "hands": hands,
            }
        )
    return result


def analyze(frames, cash_frames, p):
    """Detect output visits, then separately require new cash to hypothesize withdrawal."""
    if any(
        b["timestamp_sec"] <= a["timestamp_sec"] for a, b in zip(frames, frames[1:], strict=False)
    ):
        raise ValueError("Frame timestamps must strictly increase")
    tracks = {}
    for frame in frames:
        for hand in frame["hands"]:
            tracks.setdefault(hand["track_id"], []).append(hand)
    scale = frames[0]["counter_width_px"] if frames else 1
    contacts, markers = [], []
    for track_id, chain in tracks.items():
        index = 0
        while index < len(chain):
            start = index
            if not chain[start]["inside_output_zone"]:
                index += 1
                continue
            end = start
            while (
                end + 1 < len(chain)
                and chain[end + 1]["inside_output_zone"]
                and chain[end + 1]["timestamp_sec"] - chain[end]["timestamp_sec"]
                <= p["hand_track_max_gap_sec"]
            ):
                end += 1
            index = end + 1
            inside = chain[start : end + 1]
            if (
                len(inside) < p["contact_min_samples"]
                or inside[-1]["timestamp_sec"] - inside[0]["timestamp_sec"] < p["contact_min_sec"]
            ):
                continue
            before = [
                n
                for n in chain[:start]
                if 0 < inside[0]["timestamp_sec"] - n["timestamp_sec"] <= p["approach_history_sec"]
            ]
            approach = next(
                (n for n in reversed(before) if n["distance_widths"] >= p["approach_delta_widths"]),
                None,
            )
            leave = None
            previous = inside[-1]
            for n in chain[end + 1 :]:
                if (
                    n["timestamp_sec"] - previous["timestamp_sec"] > p["leave_max_gap_sec"]
                    or n["inside_output_zone"]
                ):
                    break
                if n["distance_widths"] >= p["leave_distance_widths"]:
                    leave = n
                    break
                previous = n
            ownership = Counter(n["owner"] for n in inside)
            owner = next(
                (
                    r
                    for r in ["cashier", "customer"]
                    if ownership[r] >= 2
                    and ownership[r] / len(inside) >= 0.6
                    and ownership["customer" if r == "cashier" else "cashier"] == 0
                ),
                "unknown",
            )
            issues = [
                "2D wrist proximity does not establish physical contact or cash extraction.",
                "Output zone includes a fixed palm/wrist allowance around the visible tray.",
            ]
            if owner == "unknown":
                issues.append("Acting hand ownership remains unknown.")
            if not approach:
                issues.append("No observed approach on the same wrist track.")
            if not leave:
                issues.append("No observed leave on the same wrist track; loss is not a leave.")
            evidence = (
                ([approach] if approach else [])
                + [inside[0], inside[-1]]
                + ([leave] if leave else [])
            )
            contact = {
                "id": f"contact-{len(contacts) + 1}",
                "hand_track_id": track_id,
                "side": inside[0]["side"],
                "timestamp_sec": inside[0]["timestamp_sec"],
                "end_sec": inside[-1]["timestamp_sec"],
                "leave_sec": leave["timestamp_sec"] if leave else None,
                "owner": owner,
                "owner_counts": dict(ownership),
                "complete_sequence": bool(approach and leave),
                "observed_features": {
                    "approach_delta_widths": approach["distance_widths"] if approach else None,
                    "inside_samples": len(inside),
                    "inside_duration_sec": inside[-1]["timestamp_sec"] - inside[0]["timestamp_sec"],
                    "min_core_distance_widths": min(n["core_distance_widths"] for n in inside),
                },
                "confidence": min(0.75, min(n["wrist_confidence"] for n in inside)),
                "wrist_evidence": [
                    {
                        "timestamp_sec": n["timestamp_sec"],
                        "frame_index": n["frame_index"],
                        "xy": n["xy"],
                        "owner": n["owner"],
                        "wrist_confidence": n["wrist_confidence"],
                    }
                    for n in evidence
                ],
                "cash_links": [],
                "withdrawal_hypothesis": False,
                "uncertainty": issues,
            }
            for kind, node in [
                ("HAND_APPROACH", approach),
                ("HAND_IN_OUTPUT_ZONE", inside[0]),
                ("HAND_LEAVE_OUTPUT_ZONE", leave),
            ]:
                if node:
                    markers.append(
                        {
                            "type": kind,
                            "timestamp_sec": node["timestamp_sec"],
                            "frame_index": node["frame_index"],
                            "contact_id": contact["id"],
                            "hand_track_id": track_id,
                            "owner": node["owner"],
                            "confidence": min(0.75, node["wrist_confidence"]),
                            "uncertainty": issues.copy(),
                        }
                    )
            if leave:
                # Match actual cash observations to wrist, without track continuity requirements.
                for cf in cash_frames:
                    ct = cf["timestamp_sec"]
                    if (
                        not leave["timestamp_sec"]
                        <= ct
                        <= leave["timestamp_sec"] + p["cash_followup_sec"]
                    ):
                        continue
                    hand = min(chain, key=lambda n: abs(n["timestamp_sec"] - ct))
                    if abs(hand["timestamp_sec"] - ct) > p["hand_track_max_gap_sec"]:
                        continue
                    for obj in cf["objects"]:
                        if obj["label"] != "cash" or obj["confidence"] < p["cash_min_confidence"]:
                            continue
                        center = [obj["center"][0] * cf["width"], obj["center"][1] * cf["height"]]
                        gap = math.dist(center, hand["xy"]) / scale
                        if gap <= p["cash_wrist_distance_widths"]:
                            contact["cash_links"].append(
                                {
                                    "timestamp_sec": ct,
                                    "frame_index": cf["frame_index"],
                                    "cash_bbox": obj["bbox"],
                                    "cash_confidence": obj["confidence"],
                                    "cash_logical_id": obj["logical_object_id"],
                                    "wrist_distance_widths": gap,
                                    "wrist_owner": hand["owner"],
                                    "wrist_xy": hand["xy"],
                                }
                            )
                prior = []
                for cf in cash_frames:
                    ct = cf["timestamp_sec"]
                    if (
                        not inside[0]["timestamp_sec"] - p["cash_existing_before_sec"]
                        <= ct
                        <= inside[0]["timestamp_sec"]
                    ):
                        continue
                    hand = min(chain, key=lambda n: abs(n["timestamp_sec"] - ct))
                    if abs(hand["timestamp_sec"] - ct) > p["hand_track_max_gap_sec"]:
                        continue
                    for obj in cf["objects"]:
                        if obj["label"] == "cash" and obj["confidence"] >= p["cash_min_confidence"]:
                            center = [
                                obj["center"][0] * cf["width"],
                                obj["center"][1] * cf["height"],
                            ]
                            if (
                                math.dist(center, hand["xy"]) / scale
                                <= p["cash_wrist_distance_widths"]
                            ):
                                prior.append(ct)
                cashier_links = [
                    link for link in contact["cash_links"] if link["wrist_owner"] == "cashier"
                ]
                contact["observed_features"]["cash_before_contact_times"] = prior
                contact["observed_features"]["cash_after_leave_frames"] = len(
                    {n["frame_index"] for n in cashier_links}
                )
                contact["withdrawal_hypothesis"] = bool(
                    contact["complete_sequence"]
                    and owner == "cashier"
                    and not prior
                    and len({n["frame_index"] for n in cashier_links}) >= p["cash_min_samples"]
                )
                if prior:
                    issues.append(
                        "Cash already visible beside this hand before contact; "
                        "new extraction is unproven."
                    )
                if not contact["cash_links"]:
                    issues.append(
                        "No cash observation linked to the leaving wrist in the follow-up window."
                    )
                elif len({n["frame_index"] for n in cashier_links}) < p["cash_min_samples"]:
                    issues.append("Too few cash observations with a reliably cashier-owned wrist.")
                if contact["withdrawal_hypothesis"]:
                    contact["confidence"] = min(0.75, contact["confidence"])
                    issues.append(
                        "Cash model can confuse empty hands or ID; "
                        "this is a diagnostic withdrawal hypothesis only."
                    )
            contacts.append(contact)
    contacts.sort(key=lambda n: n["timestamp_sec"])
    # A two-handed single visit must not create two withdrawal hypotheses.
    accepted = []
    for contact in contacts:
        if contact["withdrawal_hypothesis"]:
            if any(
                abs(contact["timestamp_sec"] - other["timestamp_sec"]) <= p["duplicate_contact_sec"]
                for other in accepted
            ):
                contact["withdrawal_hypothesis"] = False
                contact["uncertainty"].append("Suppressed duplicate simultaneous output visit.")
            else:
                accepted.append(contact)
    return {
        "contacts": contacts,
        "markers": sorted(markers, key=lambda n: n["timestamp_sec"]),
        "summary": {
            "hand_tracks": len(tracks),
            "wrist_observations": sum(map(len, tracks.values())),
            "unknown_wrist_observations": sum(
                n["owner"] == "unknown" for chain in tracks.values() for n in chain
            ),
            "output_contacts": len(contacts),
            "unknown_owner_contacts": sum(n["owner"] == "unknown" for n in contacts),
            "complete_sequences": sum(n["complete_sequence"] for n in contacts),
            "withdrawal_hypotheses": sum(n["withdrawal_hypothesis"] for n in contacts),
        },
    }
