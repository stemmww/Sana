"""Run fixed hypotheses and compare to separate review intervals, never fit them."""

import copy
from collections import Counter

from experiments.counter_occupancy import read_json, sha, write_json
from experiments.counter_sequence import analyze
from experiments.counter_sequence_data import DATA, META


def score(events, truth):
    used, rows = set(), []
    for event in events:
        if event["type"] != "PROBABLE_COUNTER_INTERACTION":
            continue
        matches = [
            n
            for n, action in enumerate(truth["actions"])
            if n not in used
            and action["hypothesis"] == event["hypothesis"]
            and action["start_sec"] <= event["timestamp_sec"] <= action["end_sec"]
        ]
        if matches:
            used.add(matches[0])
        rows.append(
            {
                "event_id": event["id"],
                "timestamp_sec": event["timestamp_sec"],
                "hypothesis": event["hypothesis"],
                "result": "TP" if matches else "FP",
                "matched_action": truth["actions"][matches[0]]["id"] if matches else None,
                "in_reviewed_negative": any(
                    a <= event["timestamp_sec"] <= b for a, b in truth["negative_intervals"]
                ),
                "confidence": event["confidence"],
            }
        )
    return {
        "true_interactions": len(truth["actions"]),
        "detected": len(used),
        "missed": len(truth["actions"]) - len(used),
        "false_events": sum(r["result"] == "FP" for r in rows),
        "negative_duration_sec": sum(b - a for a, b in truth["negative_intervals"]),
        "negative_false_events": sum(
            r["result"] == "FP" and r["in_reviewed_negative"] for r in rows
        ),
        "matched_events": rows,
        "missed_action_ids": [a["id"] for n, a in enumerate(truth["actions"]) if n not in used],
        "diagnostic_marker_counts": dict(Counter(e["type"] for e in events)),
    }


def main():
    protocol = read_json(META / "protocol.json")
    results = []
    for source in read_json(META / "sources.json"):
        data = read_json(DATA / source["id"] / "features.json")
        for mode in ["body_box_reach", "zone_motion_only"]:
            frames = copy.deepcopy(data["frames"])
            for frame in frames:
                frame["support_kind"] = mode
                frame["interaction_support"] = (
                    frame["cashier_reach_motion_proxy"]
                    if mode == "body_box_reach"
                    else (
                        frame["cashier_visible"]
                        and frame["counter_motion_fraction"] >= protocol["motion_fraction"]
                    )
                )
            prediction = analyze(frames, protocol)
            prediction["mode"] = mode
            prediction["protocol_sha256"] = sha(META / "protocol.json")
            prediction["input_features_sha256"] = sha(DATA / source["id"] / "features.json")
            prediction["source_video_sha256"] = source["video_sha256"]
            write_json(DATA / source["id"] / f"{mode}-events.json", prediction)
            # Annotations are first loaded AFTER prediction; analyze() has no metadata access.
            truth = read_json(META / "ground-truth.json")["videos"][source["id"]]
            metrics = score(prediction["events"], truth)
            marker_review = read_json(META / "marker-review.json")["videos"][source["id"]]
            metrics["false_auxiliary_markers"] = marker_review["false_markers"]
            results.append({"video": source["id"], "mode": mode, **metrics})
            print(
                source["id"],
                mode,
                {
                    k: metrics[k]
                    for k in ["detected", "missed", "false_events", "diagnostic_marker_counts"]
                },
            )
            for e in prediction["events"]:
                print(
                    round(e["timestamp_sec"], 2),
                    e["type"],
                    e.get("hypothesis", ""),
                    e["confidence"],
                )
    write_json(
        META / "results.json",
        {
            "protocol_sha256": sha(META / "protocol.json"),
            "ground_truth_sha256": sha(META / "ground-truth.json"),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
