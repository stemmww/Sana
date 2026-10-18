"""Pure experimental action proposals. No Rule Engine, filenames or label access.

Scores are evidence summaries, never calibrated probabilities. A multi-cue match
remains a hypothesis until independently validated; no business events are emitted.
"""

import math
from collections import defaultdict

import numpy as np

from experiments.counter_hand import distance


def wrist_cash(hands, cash_frames, p):
    """Associate each measured cash bbox with a unique nearby wrist in time.

    This is an observation, not proof of holding: ID/cash confusion and 2D overlap
    remain possible. Unknown ownership stays unknown. Never reconnect cash IDs.
    """
    result = []
    for frame in cash_frames:
        t = frame["timestamp_sec"]
        hf = min(hands, key=lambda f: abs(f["timestamp_sec"] - t), default=None)
        if hf is None or abs(hf["timestamp_sec"] - t) > p["cash_pose_time_tolerance_sec"]:
            continue
        scale = hf["counter_width_px"]
        for obj in frame["objects"]:
            if obj["label"] != "cash" or obj["confidence"] < p["cash_confidence"]:
                continue
            box = obj["bbox"]
            center = [(box[0] + box[2]) / 2 * hf["width"], (box[1] + box[3]) / 2 * hf["height"]]
            nearby = sorted(
                [(math.dist(center, h["xy"]) / scale, i, h) for i, h in enumerate(hf["hands"])],
                key=lambda x: x[0],
            )
            if not nearby or nearby[0][0] > p["cash_wrist_distance_widths"]:
                continue
            d, _, hand = nearby[0]
            ambiguous = len(nearby) > 1 and nearby[1][0] - d < p["cash_owner_margin_widths"]
            competing = [h for dist, _, h in nearby if dist - d < p["cash_owner_margin_widths"]]
            owners = {(h["owner"], h.get("body_track_id")) for h in competing}
            same_person = (
                len(owners) == 1
                and hand["owner"] != "unknown"
                and hand.get("body_track_id") is not None
            )
            owner_ambiguous = ambiguous and not same_person
            result.append(
                {
                    "timestamp_sec": t,
                    "frame_index": frame["frame_index"],
                    "logical_cash_id": obj["logical_object_id"],
                    "bbox": box,
                    "center_px": center,
                    "confidence": obj["confidence"],
                    "hand_track_id": hand["track_id"],
                    "wrist_xy": hand["xy"],
                    "owner": "unknown" if owner_ambiguous else hand["owner"],
                    "ambiguous": ambiguous,
                    "owner_ambiguous": owner_ambiguous,
                    "competing_hand_ids": [h["track_id"] for h in competing],
                    "owner_basis": "same_known_person"
                    if same_person
                    else "unique_wrist_or_unknown",
                    "wrist_distance_widths": d,
                    "wrist_confidence": hand["wrist_confidence"],
                }
            )
    return result


def window(t, duration, p):
    start = max(0.0, t - p["window_before_sec"])
    return start, min(duration, start + p["window_sec"])


def apparatus_proposals(hands, cash, p, duration):
    tracks = defaultdict(list)
    for frame in hands:
        for hand in frame["hands"]:
            tracks[hand["track_id"]].append(hand)
    w, h = p["dimensions"]
    scale = hands[0]["counter_width_px"]
    proposals = []
    for kind, zone, allowance in [
        ("CASH_LOADING_ACTION", "input", p["input_wrist_allowance_widths"]),
        ("CASH_EXTRACTION_ACTION", "output", 0.0),
    ]:
        polygon = [[x * w, y * h] for x, y in p["zones"][zone]]
        for identity, chain in tracks.items():
            ds = [distance(n["xy"], polygon) / scale for n in chain]
            i = 0
            while i < len(chain):
                if ds[i] > allowance:
                    i += 1
                    continue
                a = i
                while (
                    i + 1 < len(chain)
                    and ds[i + 1] <= allowance
                    and chain[i + 1]["timestamp_sec"] - chain[i]["timestamp_sec"]
                    <= p["hand_gap_sec"]
                ):
                    i += 1
                b = i
                i += 1
                if b - a + 1 < 2 or chain[b]["timestamp_sec"] - chain[a]["timestamp_sec"] < 0.05:
                    continue
                owners = [n["owner"] for n in chain[a : b + 1]]
                if "cashier" not in owners and "unknown" not in owners:
                    continue
                t = chain[a]["timestamp_sec"]
                start, end = window(t, duration, p)
                before = [j for j in range(a) if chain[j]["timestamp_sec"] >= start]
                approach = next(
                    (j for j in reversed(before) if ds[j] - ds[a] >= p["approach_delta_widths"]),
                    None,
                )
                leave = None
                for j in range(b + 1, len(chain)):
                    if (
                        chain[j]["timestamp_sec"] > end
                        or chain[j]["timestamp_sec"] - chain[j - 1]["timestamp_sec"]
                        > p["hand_gap_sec"]
                    ):
                        break
                    if ds[j] - allowance >= p["leave_delta_widths"]:
                        leave = j
                        break
                contact_end = chain[b]["timestamp_sec"]
                links = [
                    x
                    for x in cash
                    if x["hand_track_id"] == identity and start <= x["timestamp_sec"] <= end
                ]
                prior = [x for x in links if x["timestamp_sec"] < t and x["owner"] == "cashier"]
                after = [
                    x
                    for x in links
                    if leave is not None
                    and x["timestamp_sec"] >= chain[leave]["timestamp_sec"]
                    and x["owner"] == "cashier"
                ]
                selected = [chain[j] for j in [approach, a, b, leave] if j is not None]
                proposals.append(
                    {
                        "type": kind,
                        "timestamp_sec": t,
                        "contact_end_sec": contact_end,
                        "window_start_sec": start,
                        "window_end_sec": end,
                        "hand_track_id": identity,
                        "owner": "cashier"
                        if owners.count("cashier") / len(owners) >= 0.6 and "customer" not in owners
                        else "unknown",
                        "zone": zone,
                        "observations": {
                            "hands": selected,
                            "cash_before": prior,
                            "cash_after": after,
                        },
                        "features": {
                            "approach": approach is not None,
                            "leave": leave is not None,
                            "contact_samples": b - a + 1,
                            "contact_duration_sec": contact_end - t,
                            "contact_inside_window": contact_end <= end,
                            "cash_before_samples": len({x["frame_index"] for x in prior}),
                            "cash_after_samples": len({x["frame_index"] for x in after}),
                        },
                    }
                )
    return proposals


def handover_proposals(hands, cash, p, duration):
    """Candidate windows from cashier/customer wrist proximity, never from labels."""
    pairs = []
    for frame in hands:
        cashier = [h for h in frame["hands"] if h["owner"] == "cashier"]
        customer = [h for h in frame["hands"] if h["owner"] == "customer"]
        choices = [
            (math.dist(a["xy"], b["xy"]) / frame["counter_width_px"], a, b)
            for a in cashier
            for b in customer
        ]
        if choices:
            d, a, b = min(choices, key=lambda x: x[0])
            pairs.append(
                {
                    "timestamp_sec": frame["timestamp_sec"],
                    "distance": d,
                    "cashier": a,
                    "customer": b,
                    "pair": (a["track_id"], b["track_id"]),
                }
            )
    runs = []
    for pair in pairs:
        if pair["distance"] > p["handover_contact_widths"]:
            continue
        if (
            not runs
            or pair["timestamp_sec"] - runs[-1][-1]["timestamp_sec"] > p["hand_gap_sec"]
            or pair["pair"] != runs[-1][-1]["pair"]
        ):
            runs.append([])
        runs[-1].append(pair)
    result = []
    for run in runs:
        if len(run) < 2:
            continue
        t = run[0]["timestamp_sec"]
        start, end = window(t, duration, p)
        endcontact = run[-1]["timestamp_sec"]
        same = [
            r for r in pairs if r["pair"] == run[0]["pair"] and start <= r["timestamp_sec"] <= end
        ]
        before = [r for r in same if r["timestamp_sec"] < t]
        after = [r for r in same if r["timestamp_sec"] > endcontact]
        approach = any(
            r["distance"] - run[0]["distance"] >= p["approach_delta_widths"] for r in before
        )
        leave = any(r["distance"] >= p["handover_leave_widths"] for r in after)
        # Touch can begin before the actual transfer; require cashier cash before
        # contact, and positive customer cash at/after it. No absent cash inference.
        prior = [
            x
            for x in cash
            if start <= x["timestamp_sec"] < t
            and x["owner"] == "cashier"
            and x["hand_track_id"] == run[0]["pair"][0]
        ]
        following = [
            x
            for x in cash
            if endcontact <= x["timestamp_sec"] <= end
            and x["owner"] == "customer"
            and x["hand_track_id"] == run[0]["pair"][1]
        ]
        ids = {x["logical_cash_id"] for x in prior} & {x["logical_cash_id"] for x in following}
        direction = []
        for identity in ids:
            a = next(x for x in prior if x["logical_cash_id"] == identity)
            b = next(x for x in following if x["logical_cash_id"] == identity)
            target = run[0]["customer"]["xy"]
            direction.append(math.dist(a["center_px"], target) > math.dist(b["center_px"], target))
        result.append(
            {
                "type": "CASH_HANDOVER_TO_CUSTOMER",
                "timestamp_sec": t,
                "contact_end_sec": endcontact,
                "window_start_sec": start,
                "window_end_sec": end,
                "hand_track_id": list(run[0]["pair"]),
                "owner": "cashier",
                "zone": None,
                "observations": {
                    "hands": [r[k] for r in [run[0], run[-1]] for k in ["cashier", "customer"]],
                    "cash_before": prior,
                    "cash_after": following,
                },
                "features": {
                    "approach": approach,
                    "leave": leave,
                    "contact_samples": len(run),
                    "contact_duration_sec": endcontact - t,
                    "contact_inside_window": endcontact <= end,
                    "cash_before_samples": len({x["frame_index"] for x in prior}),
                    "cash_after_samples": len({x["frame_index"] for x in following}),
                    "same_logical_cash_ids": sorted(ids),
                    "cash_moves_to_recipient": any(direction),
                },
            }
        )
    return result


def roi_evidence(proposal, frames, p):
    """Measure changes at contact and after withdrawal, excluding visible wrists.

    Frames contain grayscale crops and a mask, not object/action labels. A changed
    ROI could be shadows, apparatus, money or ID; never called proof of banknotes.
    """
    zone = proposal["zone"]
    if zone is None:
        return {"available": False, "reason": "handover uses object/wrist observations"}
    selected = [
        f
        for f in frames
        if proposal["window_start_sec"] <= f["timestamp_sec"] <= proposal["window_end_sec"]
    ]
    if len(selected) < 3:
        return {"available": False, "reason": "too_few_roi_frames"}
    before = [f for f in selected if f["timestamp_sec"] < proposal["timestamp_sec"]]
    if not before:
        return {"available": False, "reason": "no_precontact_roi"}
    ref = before[0][zone]
    measurements = []
    raw_changes = []
    for frame in selected:
        item = frame[zone]
        raw_mask = ref["mask"] & item["mask"]
        raw_delta = item["gray"][raw_mask].astype(float) - ref["gray"][raw_mask].astype(float)
        raw_delta -= np.median(raw_delta)
        raw_changes.append(
            {
                "timestamp_sec": frame["timestamp_sec"],
                "changed_fraction": float(np.mean(np.abs(raw_delta) > p["roi_pixel_delta"])),
            }
        )
        mask = ref["mask"] & item["mask"] & ~ref["hands"] & ~item["hands"]
        visible = float(mask.sum() / max(1, ref["mask"].sum()))
        if visible < p["roi_min_unmasked_fraction"]:
            continue
        delta = item["gray"][mask].astype(float) - ref["gray"][mask].astype(float)
        delta -= np.median(delta)
        measurements.append(
            {
                "timestamp_sec": frame["timestamp_sec"],
                "frame_index": frame["frame_index"],
                "unmasked_fraction": visible,
                "changed_fraction": float(np.mean(np.abs(delta) > p["roi_pixel_delta"])),
            }
        )
    during = [
        m
        for m in measurements
        if proposal["timestamp_sec"] <= m["timestamp_sec"] <= proposal["contact_end_sec"]
    ]
    after = [m for m in measurements if m["timestamp_sec"] > proposal["contact_end_sec"]]
    return {
        "available": bool(measurements),
        "samples": measurements,
        "raw_changes": raw_changes,
        "raw_during_peak": max(
            [
                m["changed_fraction"]
                for m in raw_changes
                if proposal["timestamp_sec"] <= m["timestamp_sec"] <= proposal["contact_end_sec"]
            ],
            default=0.0,
        ),
        "during_peak": max([m["changed_fraction"] for m in during], default=0.0),
        "after_median": float(np.median([m["changed_fraction"] for m in after])) if after else 0.0,
        "after_samples": len(after),
        "interpretation": "appearance change only; cash/ID not discriminated",
    }


def grade(proposal, p):
    f = proposal["features"]
    roi = proposal["roi"]
    role = proposal["owner"] == "cashier"
    motion = f["approach"] and f["leave"] and f["contact_inside_window"]
    kind = proposal["type"]
    if kind == "CASH_LOADING_ACTION":
        cash = f["cash_before_samples"] >= p["cash_min_samples"]
        image = roi.get("raw_during_peak", 0) >= p["roi_change_fraction"]
        specific = image
    elif kind == "CASH_EXTRACTION_ACTION":
        cash = f["cash_after_samples"] >= p["cash_min_samples"]
        image = roi.get("raw_during_peak", 0) >= p["roi_change_fraction"]
        # Positive cash before contact contradicts an inference of a new stack
        # being retrieved. Lack of pre-contact cash alone never confirms retrieval.
        specific = image and f["cash_before_samples"] == 0
    else:
        cash = f["cash_before_samples"] >= p["cash_min_samples"] and f["cash_after_samples"] >= 1
        specific = bool(f["same_logical_cash_ids"]) and f["cash_moves_to_recipient"]
    reasons = []
    if not role:
        reasons.append("acting_hand_owner_unknown")
    if not motion:
        reasons.append("approach_contact_leave_not_continuous_in_window")
    if not cash:
        reasons.append("insufficient_directional_cash_wrist_observations")
    if not specific:
        reasons.append("roi_change_or_recipient_lineage_not_supported")
    if kind == "CASH_EXTRACTION_ACTION" and f["cash_before_samples"]:
        reasons.append("cash_already_with_actor_before_contact; extraction_origin_unproven")
    if any(
        x.get("ambiguous")
        for x in proposal["observations"]["cash_before"] + proposal["observations"]["cash_after"]
    ):
        reasons.append("same_person_ownership_does_not_resolve_which_hand_holds_cash")
    strong = role and motion and cash and specific
    reasons += [
        "cash_detector_can_confuse_ID_or_empty_hands",
        "2D_proximity_is_not_physical_possession",
        "independent_negative_action_validation_missing",
    ]
    if kind != "CASH_HANDOVER_TO_CUSTOMER":
        reasons.append("raw_ROI_change_can_be_the_hand_itself; not_independent_cash_evidence")
    if kind == "CASH_LOADING_ACTION":
        reasons.append("ROI_change_does_not_prove_cash_entered_apparatus")
    if kind == "CASH_EXTRACTION_ACTION":
        reasons.append("cash_after_contact_does_not_prove_counter_origin")
    proposal.update(
        status="hypothesis" if strong else "insufficient_evidence",
        confidence=round(0.2 * role + 0.2 * motion + 0.25 * cash + 0.2 * specific, 3),
        confidence_semantics="uncalibrated evidence score, not probability",
        uncertainty=reasons,
    )
    return proposal


def analyze(hands, cash_frames, roi_frames, p, duration):
    if not hands:
        return {"candidates": [], "cash_wrist_observations": []}
    if any(
        b["timestamp_sec"] <= a["timestamp_sec"] for a, b in zip(hands, hands[1:], strict=False)
    ):
        raise ValueError("Hands must be time ordered")
    cash = wrist_cash(hands, cash_frames, p)
    candidates = apparatus_proposals(hands, cash, p, duration) + handover_proposals(
        hands, cash, p, duration
    )
    candidates = [c for c in candidates if c["window_end_sec"] - c["window_start_sec"] >= 1.0]
    candidates.sort(key=lambda c: (c["timestamp_sec"], c["type"]))
    for i, c in enumerate(candidates):
        c["observations"]["hands"] = [
            h
            for h in c["observations"]["hands"]
            if c["window_start_sec"] <= h["timestamp_sec"] <= c["window_end_sec"]
        ]
        c["contact_end_sec"] = min(c["contact_end_sec"], c["window_end_sec"])
        c["features"]["contact_duration_sec"] = c["contact_end_sec"] - c["timestamp_sec"]
        c["id"] = f"action-{i + 1}"
        c["roi"] = roi_evidence(c, roi_frames, p)
        grade(c, p)
        c["evidence_frames"] = []
    return {
        "candidates": candidates,
        "cash_wrist_observations": cash,
        "confirmed_events": [],
        "production_integration": False,
    }
