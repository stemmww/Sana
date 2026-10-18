"""Replay measured detections; audit service boundaries and independent cash provenance.

Input filenames identify data only. All decisions live in app.events and use
geometry/observations. Existing diagnostic hypotheses retain their uncertainty.
"""

import argparse
import html
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from app.cv.analyze_events import analyze_video, sha256
from app.events.cash_provenance import active_session, analyze_cash_provenance, bind_cash_event
from app.events.config import EventConfig
from app.rules.cash_security import evaluate_cash_security
from app.rules.cash_security_schemas import CashSecurityRequest, SecurityEvent
from experiments.cash_security_run import convert_legacy
from experiments.counter_occupancy import imwrite, read_json, write_json


def security_events(legacy, provenance, auxiliary):
    events = [
        SecurityEvent.model_validate(bind_cash_event(e.model_dump(), provenance))
        if e.type.startswith("MONEY_")
        else e
        for e in convert_legacy(legacy)
    ]
    groups = {g["logical_cash_id"]: g for g in provenance["objects"]}
    aliases = provenance["aliases"]
    for value in read_json(auxiliary / "security-events.json"):
        if value["type"] in {
            "CASHIER_PRESENT",
            "CUSTOMER_ENTER",
            "CUSTOMER_PRESENT",
            "CUSTOMER_EXIT",
            "SERVICE_STARTED",
            "SERVICE_ENDED",
        }:
            continue
        # Reuse actual visual diagnostics only, with their original evidence and uncertainty.
        event = SecurityEvent.model_validate(value)
        if event.source == "synthetic_example":
            raise ValueError("Synthetic observations prohibited in real-video replay")
        event.event_id = "diagnostic-" + event.event_id
        group = groups.get(aliases.get(event.object_id))
        if group:
            event = SecurityEvent.model_validate(bind_cash_event(event.model_dump(), provenance))
        elif event.type == "MONEY_FROM_COUNTER":
            session = active_session(legacy["rule_result"]["sessions"], event.start_sec)
            event.operation_id = session["session_id"] if session else None
            event.customer_id = session["customer_id"] if session else None
        elif event.type.startswith("SAFE_"):
            session = active_session(legacy["rule_result"]["sessions"], event.start_sec)
            event.operation_id = session["session_id"] if session else None
            event.customer_id = session["customer_id"] if session else None
        events.append(event)
    return sorted(events, key=lambda e: e.end_sec)


def audit(source, output, config_path, replay):
    output.mkdir(parents=True, exist_ok=True)
    video, cache, auxiliary = (
        Path(source["video"]),
        Path(source["detections"]),
        Path(source["auxiliary"]),
    )
    metadata = read_json(cache)
    if sha256(video) != metadata["video_sha256"]:
        raise ValueError("Detection cache/video mismatch")
    if read_json(auxiliary / "legacy/events.json")["video_sha256"] != metadata["video_sha256"]:
        raise ValueError("Visual diagnostics/video mismatch")
    if replay:
        analyze_video(video, output / "legacy", config_path, replay=cache)
    legacy = read_json(output / "legacy/events.json")
    config = EventConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    if legacy["config"] != json.loads(config.model_dump_json()):
        raise ValueError("Replay config mismatch: run with --replay")
    frames = read_json(output / "legacy/track_history.json")["frames"]
    sessions = legacy["rule_result"]["sessions"]
    provenance = analyze_cash_provenance(frames, sessions, legacy["rule_request"]["events"], config)
    # Associate unapproved wrist observations with possible cash objects, never promote events.
    for contact in read_json(auxiliary / "wrist-diagnostic.json")["contacts"]:
        if not contact["withdrawal_hypothesis"]:
            continue
        for group in provenance["objects"]:
            links = [
                x
                for x in contact["cash_links"]
                if x["cash_logical_id"] in group["object_track_ids"]
            ]
            if links:
                group["counter_interaction_history"].append(
                    {
                        "type": "MONEY_FROM_COUNTER",
                        "status": "insufficient_evidence",
                        "timestamp_sec": contact["leave_sec"],
                        "confidence": contact["confidence"],
                        "evidence": links,
                        "uncertainty": contact["uncertainty"],
                    }
                )
    events = security_events(legacy, provenance, auxiliary)
    request = CashSecurityRequest(
        duration_sec=legacy["rule_request"]["duration_sec"], events=events
    )
    report = evaluate_cash_security(request)
    boundary_presence = []
    for session in sessions[1:]:
        t = session["start_sec"]
        for group in provenance["objects"]:
            table = group["table_presence_history"]
            pairs = [
                (a, b)
                for a, b in zip(table, table[1:], strict=False)
                if a["timestamp_sec"] <= t <= b["timestamp_sec"]
                and b["timestamp_sec"] - a["timestamp_sec"] <= config.max_observation_gap_sec
            ]
            if pairs:
                boundary_presence.append(
                    {
                        "session_id": session["session_id"],
                        "timestamp_sec": t,
                        "logical_cash_id": group["logical_cash_id"],
                        "source_session_id": group["source_session_id"],
                        "provenance_status": group["status"],
                        "positive_samples": pairs[0],
                    }
                )
    summary = {
        "source": source,
        "sessions": sessions,
        "customer_tracks": legacy["service_boundary"],
        "logical_cash_objects": [
            {k: v for k, v in g.items() if k != "observations"} for g in provenance["objects"]
        ],
        "cash_at_next_session": boundary_presence,
        "r05": report.model_dump(),
        "r02": [v for v in legacy["rule_result"]["violations"] if v["rule_id"] == "R02"],
        "limitations": [
            "Session start confirms presence in service interior; door crossing is earlier.",
            "Raw YOLO cash detections can confuse hands/ID; no detector weights changed.",
            "Wrist withdrawal and safe/table diagnostics retain their unvalidated status.",
        ],
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "security-request.json", request.model_dump())
    write_json(output / "security-result.json", report.model_dump())
    write_json(output / "summary.json", summary)
    lines = [
        f"{e.start_sec:.3f}-{e.end_sec:.3f} {e.type} {e.status} operation={e.operation_id} "
        f"money={e.money_group_id} confidence={e.confidence:.3f} " + "; ".join(e.uncertainty)
        for e in events
    ]
    (output / "timeline.txt").write_text("\n".join(lines), encoding="utf-8")
    # Real frames at algorithm-selected crossings and service boundaries.
    selected = set()
    for crossing in legacy["service_boundary"]["crossings"]:
        selected.update(s["timestamp_sec"] for s in crossing["samples"])
    for s in sessions:
        selected.add(s["start_sec"])
    cap = cv2.VideoCapture(str(video))
    panels = []
    for t in sorted(selected):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Cannot read evidence at {t}")
        for name, poly in config.service_boundary.zones.items():
            import numpy as np

            points = np.array([(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in poly])
            cv2.polylines(image, [points], True, (0, 220, 255), 1)
            cv2.putText(
                image, name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1
            )
        nearest = min(frames, key=lambda f: abs(f["timestamp_sec"] - t))
        for p in nearest["people"]:
            if p["role"] != "customer":
                continue
            b = p["bbox"]
            pt = (
                round((b[0] + b[2]) / 2 * image.shape[1]),
                round(
                    (b[1] + config.service_boundary.anchor_height * (b[3] - b[1])) * image.shape[0]
                ),
            )
            cv2.circle(image, pt, 4, (0, 0, 255), -1)
        imwrite(output / "evidence" / f"{round(t * metadata['fps']):05d}.jpg", image)
        thumb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((232, 424))
        panel = Image.new("RGB", (232, 448), "white")
        panel.paste(thumb, (0, 24))
        ImageDraw.Draw(panel).text((4, 4), f"{t:.3f}s", fill="black")
        panels.append(panel)
    cap.release()
    sheet = Image.new("RGB", (232 * 4, 448 * ((len(panels) + 3) // 4)), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 4) * 232, (i // 4) * 448))
    sheet.save(output / "boundary-evidence.jpg")
    # Cash examples chosen by observation history, not by scenario timestamps.
    selected_groups = [
        g
        for g in provenance["objects"]
        if g["counter_interaction_history"]
        or len(g["object_track_ids"]) > 1
        or any(b["logical_cash_id"] == g["logical_cash_id"] for b in boundary_presence)
    ]
    cap = cv2.VideoCapture(str(video))
    cash_panels = []
    for group in selected_groups:
        samples = [group["observations"][0], group["observations"][-1]]
        group["evidence_frames"] = []
        for obs in samples:
            t = obs["timestamp_sec"]
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot read cash evidence at {t}")
            h, w = frame.shape[:2]
            box = [round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(obs["bbox"])]
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
            path = (
                output
                / "evidence"
                / f"cash-{group['logical_cash_id']}-{round(t * metadata['fps'])}.jpg"
            )
            imwrite(path, frame)
            group["evidence_frames"].append(str(path))
            panel = Image.new("RGB", (232, 456), "white")
            panel.paste(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((232, 424)), (0, 32)
            )
            ImageDraw.Draw(panel).text(
                (3, 2),
                f"{group['logical_cash_id']} {t:.2f}s\n"
                f"origin={group['source_session_id']} (possible)",
                fill="black",
            )
            cash_panels.append(panel)
    cap.release()
    if cash_panels:
        sheet = Image.new("RGB", (232 * 4, 456 * ((len(cash_panels) + 3) // 4)), "white")
        for i, panel in enumerate(cash_panels):
            sheet.paste(panel, ((i % 4) * 232, (i // 4) * 456))
        sheet.save(output / "cash-evidence.jpg")
    write_json(output / "provenance.json", provenance)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    summaries = [
        audit(s, args.output / s["id"], args.config, args.replay) for s in read_json(args.sources)
    ]
    write_json(args.output / "results.json", summaries)
    body = (
        "<h1>Service boundaries and cash provenance</h1>"
        "<p>Actual cached YOLO observations. No synthetic events.</p>"
    )
    for s in summaries:
        name = s["source"]["id"]
        body += (
            f"<h2>{html.escape(name)}</h2><table><tr><th>Session</th>"
            "<th>Customer</th><th>Start</th><th>End</th></tr>"
        )
        for session in s["sessions"]:
            body += (
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(session[k]))}</td>"
                    for k in ("session_id", "customer_id", "start_sec", "end_sec")
                )
                + "</tr>"
            )
        body += f"</table><p>R02: {len(s['r02'])}; R05: {s['r05']['verdict']}</p>"
        body += (
            f'<p><a href="{name}/timeline.txt">Timeline</a> · '
            f'<a href="{name}/summary.json">Complete audit</a> · '
            f'<a href="{name}/provenance.json">Cash histories and rejected links</a></p>'
        )
        body += f'<img src="{name}/boundary-evidence.jpg" style="max-width:100%">'
        body += "<h3>Cash observation groups (not a count of real bundles)</h3>"
        body += "<table><tr><th>ID / fragments</th><th>First–last</th><th>Possible origin</th></tr>"
        for g in s["logical_cash_objects"]:
            body += (
                f"<tr><td>{g['logical_cash_id']}: "
                f"{html.escape(', '.join(g['object_track_ids']))}</td>"
                f"<td>{g['first_seen']:.2f}–{g['last_seen']:.2f}</td>"
                f"<td>{g['source_session_id']}; {g['status']}</td></tr>"
            )
        body += "</table>"
        if (args.output / name / "cash-evidence.jpg").exists():
            body += f'<img src="{name}/cash-evidence.jpg" style="max-width:100%">'
        body += (
            "<h3>Cash at second service boundary</h3><pre>"
            + html.escape(json.dumps(s["cash_at_next_session"], indent=2))
            + "</pre>"
        )
    (args.output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>'
        "body{font:16px sans-serif;max-width:1100px;margin:auto}"
        "td,th{padding:8px;border:1px solid #ccc}pre{white-space:pre-wrap}</style>" + body,
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                {
                    "video": s["source"]["id"],
                    "sessions": s["sessions"],
                    "r05": s["r05"]["verdict"],
                    "cash_objects": len(s["logical_cash_objects"]),
                }
                for s in summaries
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
