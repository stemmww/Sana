"""Auditable cash history, independent of the active service and business rules.

Recovery requires several observations on BOTH sides of a bounded gap and a
unique spatial/size/trajectory match. Shared table membership is never a match.
Session context alone is a possible origin, not proof of a counter withdrawal.
"""

from collections import defaultdict
from statistics import mean

from app.events.tracking import area, distance


def bind_cash_event(event, provenance):
    """Supply R05 with retained origin; never upgrade an unconfirmed producer."""
    value = dict(event)
    key = provenance["aliases"].get(value.get("object_id"))
    group = next((g for g in provenance["objects"] if g["logical_cash_id"] == key), None)
    if not group:
        return value
    value.update(
        operation_id=group["source_session_id"],
        customer_id=group["source_customer_id"],
        money_group_id=key,
    )
    reasons = list(dict.fromkeys(value.get("uncertainty", []) + group["uncertainty"]))
    if group["status"] != "confirmed" or reasons:
        value["status"] = "insufficient_evidence"
    value["uncertainty"] = reasons
    return value


def active_session(sessions, timestamp):
    return next(
        (
            s
            for s in sessions
            if s["start_sec"] <= timestamp and (s["end_sec"] is None or timestamp <= s["end_sec"])
        ),
        None,
    )


def recovery_cost(previous, current, config):
    n = config.reconnect_min_samples + 1
    if len(previous) < n or len(current) < n:
        return None, "insufficient_samples_on_both_sides"
    a, b = previous[-n:], current[:n]
    gap = b[0]["timestamp_sec"] - a[-1]["timestamp_sec"]
    if not 0 < gap <= config.reconnect_gap_sec:
        return None, "overlap_or_gap_exceeds_limit"
    if any(
        y["timestamp_sec"] - x["timestamp_sec"] > config.max_observation_gap_sec
        for run in (a, b)
        for x, y in zip(run, run[1:], strict=False)
    ):
        return None, "sparse_support"
    if any(x.get("association_ambiguous", False) for x in a + b):
        return None, "ambiguous_tracking"
    aa, ab = mean(area(x["bbox"]) for x in a), mean(area(x["bbox"]) for x in b)
    if max(aa / ab, ab / aa) > config.reconnect_max_area_ratio:
        return None, "incompatible_visible_size"
    dt = a[-1]["timestamp_sec"] - a[0]["timestamp_sec"]
    velocity = [(a[-1]["center"][i] - a[0]["center"][i]) / dt for i in (0, 1)]
    # Require agreement throughout the incoming window, not just one nearby bbox.
    static_error = max(distance(a[-1]["center"], x["center"]) for x in b)
    motion_error = max(
        distance(
            [
                a[-1]["center"][i] + velocity[i] * (x["timestamp_sec"] - a[-1]["timestamp_sec"])
                for i in (0, 1)
            ],
            x["center"],
        )
        for x in b
    )
    cost = min(static_error, motion_error)
    if cost > config.reconnect_distance:
        return None, "trajectory_mismatch"
    return cost, "unique_bounded_multi_observation_match"


def analyze_cash_provenance(frames, sessions, semantic_events, config):
    segments = defaultdict(list)
    for frame in frames:
        for obs in frame["objects"]:
            if obs["label"] == "cash":
                segments[obs["logical_object_id"]].append(dict(obs))
    groups, aliases, links, rejected = {}, {}, [], []
    for raw, samples in sorted(segments.items(), key=lambda item: item[1][0]["timestamp_sec"]):
        choices = []
        for key, group in groups.items():
            old = group["observations"]
            if any(
                e.get("type") == "MONEY_TO_SAFE"
                and e.get("object_id") in group["object_track_ids"]
                and e.get("status", "confirmed") == "confirmed"
                and e.get("confidence", 0) >= config.minimum_event_confidence
                and old[-1]["timestamp_sec"] <= e["end_sec"] <= samples[0]["timestamp_sec"]
                for e in semantic_events
            ):
                continue
            cost, reason = recovery_cost(old, samples, config)
            if cost is not None:
                choices.append((cost, key))
            elif 0 < samples[0]["timestamp_sec"] - old[-1]["timestamp_sec"] <= 5:
                rejected.append(
                    {
                        "from_logical_cash_id": key,
                        "to_object_track": raw,
                        "gap_sec": samples[0]["timestamp_sec"] - old[-1]["timestamp_sec"],
                        "reason": reason,
                    }
                )
        choices.sort()
        ambiguous = (
            len(choices) > 1 and choices[1][0] - choices[0][0] < config.reconnect_ambiguity_margin
        )
        if choices and not ambiguous:
            key = choices[0][1]
            links.append(
                {
                    "logical_cash_id": key,
                    "to_object_track": raw,
                    "timestamp_sec": samples[0]["timestamp_sec"],
                    "confirmed_at_sec": samples[config.reconnect_min_samples]["timestamp_sec"],
                    "gap_sec": samples[0]["timestamp_sec"]
                    - groups[key]["observations"][-1]["timestamp_sec"],
                    "reason": "unique_bounded_multi_observation_match",
                    "cost": choices[0][0],
                }
            )
            groups[key]["observations"].extend(samples)
            groups[key]["object_track_ids"].append(raw)
        else:
            key = f"logical-cash-{len(groups) + 1}"
            groups[key] = {
                "logical_cash_id": key,
                "object_track_ids": [raw],
                "observations": samples,
                "ambiguous_recovery": ambiguous,
            }
        aliases[raw] = key
    for group in groups.values():
        obs = group["observations"]
        source = active_session(sessions, obs[0]["timestamp_sec"])
        original = [
            e
            for e in semantic_events
            if e.get("object_id") in group["object_track_ids"]
            and e.get("type") == "MONEY_TRANSFER_TO_CASHIER"
            and e.get("confidence", 0) >= config.minimum_event_confidence
            and e.get("status", "confirmed") == "confirmed"
        ]
        confirmed_source = active_session(sessions, original[0]["end_sec"]) if original else None
        # An unrelated later session must not adopt an already visible stack.
        if confirmed_source and (
            not source or confirmed_source["session_id"] != source["session_id"]
        ):
            confirmed_source = None
        counter = [
            e
            for e in semantic_events
            if e.get("object_id") in group["object_track_ids"]
            and e.get("type") in {"MONEY_TO_COUNTER", "MONEY_FROM_COUNTER"}
        ]
        reasons = []
        if not confirmed_source:
            reasons.append(
                "session_context_only_not_confirmed_cash_transfer"
                if source
                else "first_seen_between_services_origin_unknown"
            )
        if group["ambiguous_recovery"]:
            reasons.append("multiple_plausible_cash_predecessors")
        source = confirmed_source or source
        group.update(
            source_session_id=source["session_id"] if source else None,
            source_customer_id=source["customer_id"] if source else None,
            first_seen=obs[0]["timestamp_sec"],
            last_seen=obs[-1]["timestamp_sec"],
            last_known_zone=obs[-1].get("zone", "unknown"),
            associated_person_role=obs[-1].get("association", "unknown")
            if obs[-1].get("association") in {"cashier", "customer"}
            else "unknown",
            counter_interaction_history=counter,
            table_presence_history=[
                {
                    "timestamp_sec": x["timestamp_sec"],
                    "bbox": x["bbox"],
                    "confidence": x["confidence"],
                    "object_track_id": x["logical_object_id"],
                }
                for x in obs
                if x.get("zone") == "table"
            ],
            observation_confidence=round(mean(x["confidence"] for x in obs), 4),
            provenance_confidence=min(e["confidence"] for e in original)
            if confirmed_source
            else 0.5
            if source
            else 0,
            status="confirmed" if confirmed_source and not reasons else "insufficient_evidence",
            uncertainty=reasons,
        )
    return {
        "objects": list(groups.values()),
        "aliases": aliases,
        "reconnections": links,
        "rejected_reconnections": rejected,
        "maximum_reconnection_gap_sec": config.reconnect_gap_sec,
    }
