"""Standalone wrist experiment runner; reference labels are used only after inference."""

from experiments.counter_hand import analyze, observations
from experiments.counter_hand_data import DATA, META
from experiments.counter_occupancy import ROOT, read_json, sha, write_json


def score(prediction, truth):
    actions = [a for a in truth["actions"] if a["hypothesis"] == "possible_withdrawal"]
    used, decisions = set(), []
    for contact in prediction["contacts"]:
        if not contact["withdrawal_hypothesis"]:
            continue
        matches = [
            i
            for i, a in enumerate(actions)
            if i not in used
            and a["start_sec"] <= contact["end_sec"]
            and contact["timestamp_sec"] <= a["end_sec"]
        ]
        if matches:
            used.add(matches[0])
        decisions.append(
            {
                "contact_id": contact["id"],
                "result": "TP" if matches else "FP",
                "timestamp_sec": contact["timestamp_sec"],
                "leave_sec": contact["leave_sec"],
            }
        )
    return {
        "real_withdrawals": len(actions),
        "detected": len(used),
        "missed": len(actions) - len(used),
        "false_withdrawals": sum(d["result"] == "FP" for d in decisions),
        "decisions": decisions,
        "cash_linked_contacts": sum(bool(c["cash_links"]) for c in prediction["contacts"]),
        "negative_duration_sec": sum(b - a for a, b in truth["negative_intervals"]),
        **prediction["summary"],
    }


def main():
    p = read_json(META / "protocol.json")
    results = []
    for source in read_json(META / "sources.json"):
        out = DATA / source["id"]
        poses = read_json(out / "poses.json")
        roles = read_json(ROOT / source["cache"] / "track_history.json")
        cash = read_json(ROOT / "data/counter_sequence_v1" / source["id"] / "features.json")
        frames = observations(poses["frames"], roles["frames"], p)
        write_json(
            out / "hands.json",
            {
                "frames": frames,
                "protocol_sha256": sha(META / "protocol.json"),
                "poses_sha256": sha(out / "poses.json"),
                "role_cache_sha256": sha(ROOT / source["cache"] / "track_history.json"),
            },
        )
        predicted = analyze(frames, cash["frames"], p)
        predicted.update(
            {
                "protocol_sha256": sha(META / "protocol.json"),
                "decision_code_sha256": sha(ROOT / "experiments/counter_hand.py"),
                "source_sha256": source["video_sha256"],
                "cash_features_sha256": sha(
                    ROOT / "data/counter_sequence_v1" / source["id"] / "features.json"
                ),
            }
        )
        write_json(out / "predictions.json", predicted)
        # Ground truth is loaded only after all decisions for this video have been made.
        truth = read_json(ROOT / "datasets/counter_sequence_v1/ground-truth.json")["videos"][
            source["id"]
        ]
        metrics = score(predicted, truth)
        results.append({"video": source["id"], **metrics})
        print(source["id"], metrics, flush=True)
        for c in predicted["contacts"]:
            print(
                c["id"],
                round(c["timestamp_sec"], 2),
                round(c["end_sec"], 2),
                c["leave_sec"],
                c["owner"],
                "complete",
                c["complete_sequence"],
                "cash",
                len(c["cash_links"]),
                "withdrawal",
                c["withdrawal_hypothesis"],
                flush=True,
            )
    write_json(
        META / "results.json",
        {
            "results": results,
            "protocol_sha256": sha(META / "protocol.json"),
            "ground_truth_sha256": sha(ROOT / "datasets/counter_sequence_v1/ground-truth.json"),
            "reference_labels": (
                "Previous assistant visual review, not independent human gold labels"
            ),
        },
    )


if __name__ == "__main__":
    main()
