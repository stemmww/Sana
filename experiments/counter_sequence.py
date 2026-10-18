"""Pure diagnostic sequence hypotheses. No filenames, ground truth or business events.

The returned confidence is a bounded heuristic evidence score, not a probability.
Existing raw/logical tracks are retained: cross-fragment links are hypotheses only.
"""

from __future__ import annotations

import math

TYPES = {
    "CASH_APPROACHED_COUNTER",
    "CASH_DISAPPEARED_NEAR_COUNTER",
    "CASH_REAPPEARED_NEAR_COUNTER",
    "PROBABLE_COUNTER_INTERACTION",
}


def analyze(frames, parameters):
    p = parameters
    if not frames:
        return {"events": [], "chains": [], "rejected_hypotheses": []}
    if any(
        b["timestamp_sec"] <= a["timestamp_sec"] for a, b in zip(frames, frames[1:], strict=False)
    ):
        raise ValueError("Frame times must increase")
    chains = []
    for frame in frames:
        used = set()
        eligible = [
            o
            for o in frame["objects"]
            if o["label"] == "cash"
            and o["confidence"] >= p["cash_confidence"]
            and not o["inside_counter"]
        ]
        for obs in sorted(eligible, key=lambda o: o["confidence"], reverse=True):
            node = {
                "timestamp_sec": frame["timestamp_sec"],
                "frame_index": frame["frame_index"],
                "object": obs,
            }
            choices = []
            for n, chain in enumerate(chains):
                last = chain[-1]
                gap = node["timestamp_sec"] - last["timestamp_sec"]
                distance = math.dist(
                    obs["center_counter_units"], last["object"]["center_counter_units"]
                )
                if (
                    n not in used
                    and 0 < gap <= p["max_fragment_gap_sec"]
                    and distance <= p["max_fragment_distance_widths"]
                ):
                    same = obs["logical_object_id"] == last["object"]["logical_object_id"]
                    choices.append((0 if same else 1, distance, n))
            choices.sort()
            ambiguous = (
                len(choices) > 1
                and choices[0][0] == choices[1][0]
                and abs(choices[0][1] - choices[1][1]) < 0.05
            )
            if choices and not ambiguous:
                idx = choices[0][2]
                chains[idx].append(node)
            else:
                idx = len(chains)
                node["hypothesis_association_ambiguous"] = ambiguous
                chains.append([node])
            used.add(idx)

    events, rejected = [], []

    def event(kind, t, nodes, supporting_frames=(), **extra):
        objects = [n["object"] for n in nodes]
        logical = list(dict.fromkeys(o["logical_object_id"] for o in objects))
        issues = [
            (
                "Cash class can be wrong (including ID); cash-only detector is not independently "
                "reliable."
            ),
            "Cashier ownership is a body-box proximity proxy; no hand/keypoint/contact detector.",
        ]
        if len(logical) > 1:
            issues.append(
                "Different logical tracks are linked only as a sequence hypothesis, not a "
                "recovered object identity."
            )
        if any(
            n.get("hypothesis_association_ambiguous") or n["object"].get("association_ambiguous")
            for n in nodes
        ):
            issues.append("At least one object association has competing geometric matches.")
        if any(f.get("actor_ambiguous") for f in supporting_frames):
            issues.append(
                "Another person's box also reaches the counter neighborhood; acting hand is "
                "ambiguous."
            )
        if kind == "CASH_DISAPPEARED_NEAR_COUNTER":
            issues.append(
                "Detection absence can be blur, occlusion or a model miss; it is not proof "
                "of deposit."
            )
            issues.append(
                "Absence refers to eligible outside-counter continuation; internal-counter "
                "detections are deliberately unused."
            )
        if kind == "CASH_REAPPEARED_NEAR_COUNTER":
            issues.append(
                "Reappearance does not establish that this is the same bundle or that it "
                "came from the machine."
            )
        if kind == "PROBABLE_COUNTER_INTERACTION":
            issues += [
                "Image motion and body-box reach do not prove physical hand contact.",
                (
                    "No banknote detection inside the machine is used; placement/removal remains "
                    "unconfirmed."
                ),
            ]
        zone_only = any(f.get("support_kind") == "zone_motion_only" for f in supporting_frames)
        if zone_only:
            issues.append(
                "Only counter-local motion plus a visible cashier supports interaction; "
                "motion is NOT assigned to the cashier's hand."
            )
        refs = [
            {
                "frame_index": n["frame_index"],
                "timestamp_sec": n["timestamp_sec"],
                "role": "cash_detection",
            }
            for n in nodes
        ]
        refs += [
            {
                "frame_index": f["frame_index"],
                "timestamp_sec": f["timestamp_sec"],
                "role": "counter_motion_or_gap",
            }
            for f in supporting_frames
        ]
        # Keep the full list in JSON; evidence gallery chooses representative entries only.
        refs = list({(r["frame_index"], r["role"]): r for r in refs}.values())
        feature = {
            "cash_detections": [
                {
                    "timestamp_sec": n["timestamp_sec"],
                    "frame_index": n["frame_index"],
                    "confidence": n["object"]["confidence"],
                    "bbox": n["object"]["bbox"],
                    "counter_distance_widths": n["object"]["counter_distance_widths"],
                    "proxy_owner": n["object"]["proxy_owner"],
                    "logical_object_id": n["object"]["logical_object_id"],
                    "original_association": n["object"].get("association"),
                    "original_stable_association": n["object"].get("stable_association"),
                }
                for n in nodes
            ],
            "counter_motion": [
                {
                    "timestamp_sec": f["timestamp_sec"],
                    "frame_index": f["frame_index"],
                    "fraction": f["counter_motion_fraction"],
                    "cashier_box_gap_widths": f["cashier_box_gap_widths"],
                    "reach_motion_proxy": f["cashier_reach_motion_proxy"],
                    "actor_ambiguous": f["actor_ambiguous"],
                    "support_kind": f.get("support_kind", "body_box_reach"),
                }
                for f in supporting_frames
            ],
            "logical_track_ids": logical,
        }
        score = float(extra.pop("confidence"))
        if zone_only and kind == "PROBABLE_COUNTER_INTERACTION":
            score = min(score, 0.55)
        e = {
            "id": f"diagnostic-{len(events) + 1}",
            "type": kind,
            "timestamp_sec": float(t),
            "observed_features": feature,
            "confidence": score,
            "confidence_kind": "heuristic_evidence_score_not_probability",
            "evidence_frames": refs,
            "uncertainty_reasons": issues,
            "production_event": False,
            **extra,
        }
        events.append(e)
        return e

    approaches = []
    for chain in chains:
        for k, node in enumerate(chain):
            t, obj = node["timestamp_sec"], node["object"]
            history = [n for n in chain[: k + 1] if t - n["timestamp_sec"] <= p["history_sec"]]
            held = [n for n in history if n["object"]["proxy_owner"] == "cashier"]
            if (
                len(held) < p["cashier_confirmation_samples"]
                or obj["counter_distance_widths"] > p["near_counter_widths"]
            ):
                continue
            far = max(history, key=lambda n: n["object"]["counter_distance_widths"])
            reduction = far["object"]["counter_distance_widths"] - obj["counter_distance_widths"]
            if reduction < p["min_approach_widths"] or far["timestamp_sec"] >= t:
                continue
            near = [
                n
                for n in chain[k:]
                if n["object"]["counter_distance_widths"] <= p["near_counter_widths"]
            ]
            # A later detected movement away cancels a disappearance interpretation.
            last = near[-1]
            moved_away = any(n["timestamp_sec"] > last["timestamp_sec"] for n in chain)
            a = event(
                "CASH_APPROACHED_COUNTER",
                t,
                history,
                confidence=0.60,
                distance_reduction_widths=reduction,
                observation_status="candidate_cash_approach",
            )
            record = {"approach": a, "history": history, "last": last, "probable": None}
            approaches.append(record)
            gap_frames = [
                f
                for f in frames
                if last["timestamp_sec"]
                < f["timestamp_sec"]
                <= last["timestamp_sec"] + p["interaction_timeout_sec"]
            ]
            gap_confirm = next(
                (
                    f
                    for f in gap_frames
                    if f["timestamp_sec"] - last["timestamp_sec"] >= p["disappearance_sec"]
                ),
                None,
            )
            if moved_away or gap_confirm is None:
                rejected.append(
                    {
                        "anchor_event_id": a["id"],
                        "reason": (
                            "Cash visibly moved away, or clip ended before a disappearance "
                            "interval."
                        ),
                    }
                )
                break
            # A new competing cash detection near the counter blocks a global disappearance claim.
            competing = [
                o
                for f in gap_frames
                if f["timestamp_sec"] <= gap_confirm["timestamp_sec"]
                for o in f["objects"]
                if o["label"] == "cash"
                and o["confidence"] >= p["cash_confidence"]
                and not o["inside_counter"]
                and o["counter_distance_widths"] <= p["near_counter_widths"]
            ]
            if competing:
                rejected.append(
                    {
                        "anchor_event_id": a["id"],
                        "reason": "Competing visible cash nearby: absence is not established.",
                    }
                )
                break
            gaps = [f for f in gap_frames if f["timestamp_sec"] <= gap_confirm["timestamp_sec"]]
            disappeared = event(
                "CASH_DISAPPEARED_NEAR_COUNTER",
                gap_confirm["timestamp_sec"],
                [last],
                gaps,
                confidence=0.40,
                anchor_event_id=a["id"],
                last_visible_sec=last["timestamp_sec"],
            )
            record["disappeared"] = disappeared
            context = [
                f
                for f in frames
                if t - 0.5 <= f["timestamp_sec"] <= last["timestamp_sec"] + p["motion_context_sec"]
                and f.get("interaction_support", f["cashier_reach_motion_proxy"])
            ]
            if len(context) >= 2:
                score = 0.65 if any(f["actor_ambiguous"] for f in context) else 0.70
                probable = event(
                    "PROBABLE_COUNTER_INTERACTION",
                    context[0]["timestamp_sec"],
                    history + [last],
                    context,
                    confidence=score,
                    hypothesis="possible_deposit",
                    anchor_event_id=a["id"],
                    decision_available_at_sec=max(
                        gap_confirm["timestamp_sec"], context[1]["timestamp_sec"]
                    ),
                )
                record["probable"] = probable
            else:
                rejected.append(
                    {
                        "anchor_event_id": a["id"],
                        "reason": (
                            "Approach and disappearance lack repeated cashier reach/motion "
                            "support; no interaction inferred."
                        ),
                    }
                )
            break

    for record in approaches:
        if "disappeared" not in record:
            continue
        last = record["last"]
        cutoff = min(last["timestamp_sec"] + p["max_cycle_sec"], frames[-1]["timestamp_sec"])
        starts = [
            c[0]
            for c in chains
            if last["timestamp_sec"] + p["min_reappearance_gap_sec"]
            <= c[0]["timestamp_sec"]
            <= cutoff
            and c[0]["object"]["counter_distance_widths"] <= p["reappear_counter_widths"]
        ]
        if not starts:
            rejected.append(
                {
                    "anchor_event_id": record["approach"]["id"],
                    "reason": (
                        "No eligible reappearance before end of clip; no withdrawal inferred."
                    ),
                }
            )
            continue
        first = min(starts, key=lambda n: n["timestamp_sec"])
        t = first["timestamp_sec"]
        following = next(c for c in chains if c[0] is first)
        context = [
            f
            for f in frames
            if t - p["motion_context_sec"] <= f["timestamp_sec"] <= t + 0.5
            and f.get("interaction_support", f["cashier_reach_motion_proxy"])
        ]
        held = [
            n
            for n in following
            if 0 <= n["timestamp_sec"] - t <= p["cashier_confirmation_sec"]
            and n["object"]["proxy_owner"] == "cashier"
        ]
        held_end = held[-1]["timestamp_sec"] if held else t
        subsequent = [
            n
            for n in following
            if held_end < n["timestamp_sec"] <= t + p["customer_followup_sec"]
            and n["object"]["proxy_owner"] == "customer"
        ]
        reappeared = event(
            "CASH_REAPPEARED_NEAR_COUNTER",
            t,
            [first] + held + subsequent,
            context,
            confidence=0.45,
            anchor_event_id=record["approach"]["id"],
            gap_sec=t - last["timestamp_sec"],
            subsequent_cashier_samples=len(held),
            subsequent_customer_association=len(held) >= 2 and len(subsequent) >= 2,
            followup_observed_until_sec=max([t] + [n["timestamp_sec"] for n in held + subsequent]),
        )
        outward = max(
            (
                n["object"]["counter_distance_widths"] - first["object"]["counter_distance_widths"]
                for n in held
            ),
            default=0,
        )
        if (
            record["probable"] is None
            or len(context) < 2
            or len(held) < p["cashier_confirmation_samples"]
            or outward < p["min_outward_widths"]
        ):
            rejected.append(
                {
                    "anchor_event_id": reappeared["id"],
                    "reason": (
                        "No supported deposit history, repeated reach/motion, outward motion, or "
                        "subsequent cashier association."
                    ),
                    "motion_samples": len(context),
                    "cashier_samples": len(held),
                    "outward_widths": outward,
                }
            )
            continue
        probable = event(
            "PROBABLE_COUNTER_INTERACTION",
            context[0]["timestamp_sec"],
            [first] + held + subsequent,
            context,
            confidence=0.65 if any(f["actor_ambiguous"] for f in context) else 0.70,
            hypothesis="possible_withdrawal",
            anchor_event_id=reappeared["id"],
            deposit_hypothesis_id=record["probable"]["id"],
            outward_distance_widths=outward,
            decision_available_at_sec=max(
                held[p["cashier_confirmation_samples"] - 1]["timestamp_sec"],
                context[1]["timestamp_sec"],
            ),
            subsequent_customer_association=len(subsequent) >= 2,
        )
        probable["uncertainty_reasons"].append(
            "Bundle identity across the gap and any return/split are not established."
        )

    return {
        "events": sorted(events, key=lambda e: (e["timestamp_sec"], e["id"])),
        "chains": [
            {
                "id": n,
                "samples": [
                    {
                        "timestamp_sec": v["timestamp_sec"],
                        "frame_index": v["frame_index"],
                        "logical_object_id": v["object"]["logical_object_id"],
                        "proxy_owner": v["object"]["proxy_owner"],
                        "counter_distance_widths": v["object"]["counter_distance_widths"],
                    }
                    for v in c
                ],
            }
            for n, c in enumerate(chains)
        ],
        "rejected_hypotheses": rejected,
    }
