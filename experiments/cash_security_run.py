"""Build honest security timelines from cached real observations, with explicit unknowns."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import cv2
import numpy as np

from app.events.cash_security import CashSecurityEventDetector
from app.rules.cash_security import evaluate_cash_security
from app.rules.cash_security_schemas import CashSecurityRequest, SecurityEvent
from app.services.person_roles import point_in_polygon
from experiments.cash_security_video import DATA, META
from experiments.counter_hand import analyze, observations
from experiments.counter_occupancy import ROOT, imread, imwrite, read_json, write_json


def operation_at(sessions, customer, t):
    return next(
        (
            s["session_id"]
            for s in sessions
            if s["customer_id"] == customer
            and s["start_sec"] <= t
            and (s["end_sec"] is None or t <= s["end_sec"])
        ),
        None,
    )


def convert_legacy(doc):
    result = []
    for raw in doc["rule_request"]["events"]:
        event = SecurityEvent(
            **raw,
            operation_id=operation_at(
                doc["rule_result"]["sessions"], raw.get("customer_id"), raw["end_sec"]
            ),
        )
        result.append(event)
    lookup = {e.event_id: e for e in result}
    for derived in doc["rule_result"]["derived_events"]:
        if derived["type"] not in {"SERVICE_STARTED", "SERVICE_ENDED"}:
            continue
        trigger = lookup[derived["trigger_event_id"]]
        result.append(
            SecurityEvent(
                event_id=f"derived-{derived['type']}-{trigger.event_id}",
                type=derived["type"],
                start_sec=trigger.start_sec,
                end_sec=trigger.end_sec,
                source="local_tracker",
                confidence=trigger.confidence,
                sample_count=trigger.sample_count,
                operation_id=derived["session_id"],
                customer_id=derived["customer_id"],
            )
        )
    return result


def candidate(kind, nodes, number, **kwargs):
    times = sorted({n["timestamp_sec"] for n in nodes})
    return SecurityEvent(
        event_id=f"security-{number}",
        type=kind,
        start_sec=times[0],
        end_sec=times[-1],
        status="insufficient_evidence",
        source="local_visual",
        sample_count=len(times),
        confidence=min(0.70, float(np.mean([n.get("confidence", 0.5) for n in nodes]))),
        evidence_times=times,
        uncertainty=["diagnostic_candidate_not_validated"],
        **kwargs,
    )


def main():
    p = read_json(ROOT / "datasets/counter_hand_v1/protocol.json")
    config = read_json(ROOT / "config/cash_security_camera.json")
    totals = []
    for source in read_json(META / "sources.json"):
        out = DATA / source["id"]
        legacy = read_json(out / "legacy/events.json")
        tracks = read_json(out / "legacy/track_history.json")["frames"]
        poses = read_json(out / "poses.json")
        hands = observations(poses["frames"], tracks, p)
        write_json(out / "hands.json", {"frames": hands})
        # Generic wrist diagnostic reuses its frozen decisions. No business-event promotion.
        cash_frames = []
        for frame in tracks:
            cash_frames.append(
                {
                    "timestamp_sec": frame["timestamp_sec"],
                    "frame_index": round(frame["timestamp_sec"] * poses["fps"]),
                    "width": 464,
                    "height": 848,
                    "objects": frame["objects"],
                }
            )
        wrist = analyze(hands, cash_frames, p)
        write_json(out / "wrist-diagnostic.json", wrist)
        events = convert_legacy(legacy)
        detector = CashSecurityEventDetector(config, events)

        def reserve_evidence(event, output=out, fps=poses["fps"]):
            times = event.evidence_times
            event.evidence_frames = [
                str((output / "evidence" / f"{round(t * fps):05d}.jpg").relative_to(ROOT))
                for t in sorted({times[0], times[len(times) // 2], times[-1]})
            ]
            return event

        for contact in wrist["contacts"]:
            if not contact["withdrawal_hypothesis"]:
                continue
            nodes = contact["wrist_evidence"]
            e = candidate(
                "MONEY_FROM_COUNTER",
                nodes,
                len(events) + 1,
                observation_basis="wrist approach/contact/leave + later cash; experimental",
                operation_id=None,
                customer_id=None,
            )
            e.uncertainty += ["unapproved_wrist_hypothesis", "cash_origin_group_not_confirmed"]
            events.append(e)
        # Stationary cash seen on TABLE_ZONE. A cash track is not proof of deposit lineage.
        series = defaultdict(list)
        for frame in tracks:
            for obj in frame["objects"]:
                if (
                    obj["label"] == "cash"
                    and not obj.get("association_ambiguous")
                    and point_in_polygon(obj["center"], config["zones"]["table"])
                ):
                    series[obj["logical_object_id"]].append(obj)
        table_candidates = []
        for identity, nodes in series.items():
            runs = []
            for node in nodes:
                if (
                    not runs
                    or node["timestamp_sec"] - runs[-1][-1]["timestamp_sec"] > 0.8
                    or math.dist(node["center"], runs[-1][0]["center"]) > 0.015
                ):
                    runs.append([])
                runs[-1].append(node)
            for run in runs:
                if len(run) < 3 or run[-1]["timestamp_sec"] - run[0]["timestamp_sec"] < 0.5:
                    continue
                clear = True
                for node in run:
                    nearest = min(
                        hands, key=lambda f: abs(f["timestamp_sec"] - node["timestamp_sec"])
                    )
                    reliable_hands = [h for h in nearest["hands"] if h["owner"] != "unknown"]
                    if not reliable_hands or any(
                        math.dist(h["xy"], [node["center"][0] * 464, node["center"][1] * 848]) < 25
                        for h in reliable_hands
                    ):
                        clear = False
                e = candidate(
                    "MONEY_LEFT_ON_TABLE",
                    run,
                    len(events) + 1,
                    object_id=identity,
                    operation_id=None,
                    customer_id=None,
                    money_group_id=None,
                )
                e.uncertainty += [
                    "stationary_image_position_is_not_proof_of_surface_contact",
                    "no_confirmed_link_to_previous_counter_withdrawal",
                ]
                event = detector.observe(
                    {
                        "event": reserve_evidence(e),
                        "facts": {"cash_visible", "stationary_on_table"}
                        | ({"hand_clear"} if clear else set()),
                        "object_centers": [n["center"] for n in run],
                        "association_reliable": False,
                        "producer_validated": False,
                        "producer": "local_visual",
                    }
                )
                events.append(event)
                table_candidates.append(
                    {
                        "event_id": event.event_id,
                        "logical_track": identity,
                        "start_sec": e.start_sec,
                        "end_sec": e.end_sec,
                        "samples": len(run),
                        "hand_clear_proxy": clear,
                        "mean_cash_confidence": float(np.mean([n["confidence"] for n in run])),
                    }
                )
        # Compare the door to an independently calibrated closed-door reference.
        # A changed patch can be a hand or shadow; proposals always remain insufficient.
        reference = imread(ROOT / "data/counter_hand_v1/calibration-raw.jpg")
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mask = np.zeros(gray_ref.shape, np.uint8)
        polygon = np.rint(np.array(config["zones"]["safe_door"]) * [464, 848]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
        mask = mask.astype(bool)
        cap = cv2.VideoCapture(source["video"])
        safe_samples, groups, current, previous = [], [], [], False
        for frame in cash_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame["frame_index"])
            ok, image = cap.read()
            if not ok:
                raise ValueError("Unreadable evidence frame")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            delta = gray[mask] - gray_ref[mask]
            delta -= np.median(delta)
            fraction = float(np.mean(np.abs(delta) > 25))
            changed = fraction >= 0.15
            node = {
                "timestamp_sec": frame["timestamp_sec"],
                "frame_index": frame["frame_index"],
                "changed_fraction": fraction,
                "confidence": 0.45,
            }
            safe_samples.append(node)
            if changed != previous:
                if current:
                    groups.append((previous, current))
                current = []
            current.append(node)
            previous = changed
        if current:
            groups.append((previous, current))
        had_change = False
        for changed, nodes in groups:
            if len(nodes) < 3 or nodes[-1]["timestamp_sec"] - nodes[0]["timestamp_sec"] < 0.5:
                continue
            if changed or had_change:
                kind = "SAFE_OPENED" if changed else "SAFE_CLOSED"
                short = nodes[:4]
                e = candidate(kind, short, len(events) + 1, safe_id=config["safe_id"])
                e.uncertainty += [
                    "door_patch_difference_can_be_occlusion_or_lighting",
                    "direction_and_interior_not_confirmed",
                ]
                events.append(
                    detector.observe(
                        {
                            "event": reserve_evidence(e),
                            "facts": {
                                "region_changed" if changed else "reference_appearance_returned"
                            },
                            "association_reliable": False,
                            "producer_validated": False,
                            "producer": "local_visual",
                        }
                    )
                )
                had_change = changed
        write_json(out / "safe-image-observations.json", safe_samples)
        # Attach real evidence to all timed candidates and real semantic events.
        for e in events:
            if not e.evidence_times:
                e.evidence_times = [e.start_sec, e.end_sec]
            chosen = [
                e.evidence_times[0],
                e.evidence_times[len(e.evidence_times) // 2],
                e.evidence_times[-1],
            ]
            e.evidence_frames = []
            for t in sorted(set(chosen)):
                index = round(t * poses["fps"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    raise ValueError("Missing evidence")
                path = out / "evidence" / f"{index:05d}.jpg"
                imwrite(path, image)
                e.evidence_frames.append(str(path.relative_to(ROOT)))
        cap.release()
        events.sort(key=lambda e: e.end_sec)
        request = CashSecurityRequest(duration_sec=poses["duration_sec"], events=events)
        report = evaluate_cash_security(request)
        write_json(out / "security-request.json", request.model_dump())
        write_json(out / "security-result.json", report.model_dump())
        write_json(out / "security-events.json", [e.model_dump() for e in events])
        counts = Counter(e.type for e in events if e.status == "confirmed")
        status = {
            kind: {
                "confirmed": counts[kind],
                "status": "confirmed" if counts[kind] else "insufficient_evidence",
            }
            for kind in [
                "SAFE_OPENED",
                "MONEY_TO_SAFE",
                "SAFE_CLOSED",
                "SAFE_LOCKED_VISUAL",
                "MONEY_LEFT_ON_TABLE",
                "ID_RETURNED_TO_CUSTOMER",
            ]
        }
        status["MONEY_TO_SAFE"]["reason"] = (
            "No positive cash crossing and visible placement inside safe with confirmed lineage"
        )
        status["SAFE_LOCKED_VISUAL"]["reason"] = (
            "No validated key/lock visual action detector; closing does not establish locking"
        )
        status["ID_RETURNED_TO_CUSTOMER"]["reason"] = (
            "Loaded detector has no id_card class; no card track or confirmed return"
        )
        write_json(out / "capability-status.json", status)
        timeline = [
            f"{e.end_sec:07.3f}s {e.type} [{e.status}] confidence={e.confidence:.3f} "
            f"interval={e.start_sec:.3f}..{e.end_sec:.3f} | {'; '.join(e.uncertainty)}"
            for e in events
        ]
        timeline.append(f"R05: {report.verdict}; violations={len(report.violations)}")
        timeline += [
            f"UNOBSERVED {kind}: insufficient_evidence"
            for kind, value in status.items()
            if not value["confirmed"] and not any(e.type == kind for e in events)
        ]
        (out / "timeline.txt").write_text("\n".join(timeline) + "\n", encoding="utf-8")
        raw = read_json(out / "input-detections.json")
        id_count = sum(o["label"] == "id_card" for f in raw["frames"] for o in f["objects"])
        summary = {
            "video": source["id"],
            "verdict": report.verdict,
            "violations": len(report.violations),
            "confirmed_events": dict(counts),
            "id_detections": id_count,
            "id_tracks": len(
                {
                    o["logical_object_id"]
                    for f in tracks
                    for o in f["objects"]
                    if o["label"] == "id_card"
                }
            ),
            "id_event_counts": {
                k: counts[k]
                for k in ["ID_TRANSFER_TO_CASHIER", "ID_CHECK", "ID_RETURNED_TO_CUSTOMER"]
            },
            "wrist_withdrawal_hypotheses": wrist["summary"]["withdrawal_hypotheses"],
            "table_candidates": table_candidates,
            "safe_candidate_counts": dict(
                Counter(e.type for e in events if e.type.startswith("SAFE_"))
            ),
            "second_service_confirmed": len(legacy["rule_result"]["sessions"]) > 1,
            "first_customer_id": legacy["rule_result"]["sessions"][0]["customer_id"],
        }
        totals.append(summary)
        print(summary, flush=True)
    write_json(META / "results.json", totals)


if __name__ == "__main__":
    main()
