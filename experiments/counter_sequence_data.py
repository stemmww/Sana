"""Prepare read-only inputs and image motion for an isolated sequence diagnostic."""

from __future__ import annotations

import argparse
import copy

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experiments.counter_occupancy import ROOT, read_json, sha, write_json

DATA = ROOT / "data/counter_sequence_v1"
META = ROOT / "datasets/counter_sequence_v1"


def prepare():
    if (META / "sources.json").exists():
        raise FileExistsError("Experiment already prepared")
    DATA.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    paths = set(ROOT.glob("app/**/*.py")) | set(ROOT.glob("config/**/*"))
    paths |= {p for p in ROOT.rglob("*.pt") if ".venv" not in p.parts}
    paths |= set(ROOT.glob("data/events/**/*.json")) | set(ROOT.glob("data/events/**/*.txt"))
    paths |= set(ROOT.glob("data/events/**/debug.mp4"))
    write_json(
        META / "protected-sha256.json",
        {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if p.is_file()},
    )
    labels = read_json(ROOT / "datasets/cash_id_v2/annotations.json")["items"]
    training_sources = read_json(ROOT / "data/cash_id_v2/scouted-training-sources.json")
    sources = []
    for number in [3, 2, 1, 5]:
        key = f"x{number}"
        if number in {3, 2}:
            original = next(i for i in labels if i["id"].startswith(f"v{number:02d}_"))
        else:
            original = next(i for i in training_sources if i["video_number"] == number)
        source = {"id": key, "video": original["video"], "video_sha256": original["video_sha256"]}
        assert sha(source["video"]) == source["video_sha256"]
        if number != 5:
            source["cache"] = f"data/events/audit/after/{key}"
            source["older_debug"] = f"data/events/{key}/debug.mp4"
            source["rejected_candidate_cache"] = f"data/events/audit/candidate/{key}"
        else:
            source["cache"] = "data/counter_sequence_v1/x5"
        sources.append(source)
    write_json(META / "sources.json", sources)
    # Fixed diagnostic rules, expressed in counter widths and elapsed durations.
    write_json(
        META / "protocol.json",
        {
            "version": 1,
            "cash_confidence": 0.25,
            "near_counter_widths": 0.75,
            "reappear_counter_widths": 1.0,
            "min_approach_widths": 0.25,
            "history_sec": 3.0,
            "max_fragment_gap_sec": 1.5,
            "max_fragment_distance_widths": 1.0,
            "disappearance_sec": 0.5,
            "min_reappearance_gap_sec": 1.0,
            "motion_gray_delta": 18,
            "motion_fraction": 0.02,
            "reach_box_gap_widths": 0.35,
            "motion_context_sec": 1.5,
            "interaction_timeout_sec": 4.0,
            "cashier_confirmation_samples": 2,
            "cashier_confirmation_sec": 2.0,
            "max_cycle_sec": 30.0,
            "episode_cooldown_sec": 2.0,
            "customer_followup_sec": 6.0,
            "min_outward_widths": 0.25,
            "confidence_semantics": (
                "Heuristic evidence score, NOT a calibrated probability or production acceptance."
            ),
            "decision_inputs": (
                "relative time, cash geometry/confidence/track IDs, person roles/boxes, zone "
                "motion; never filenames or annotation times"
            ),
            "cash_inside_counter_used": False,
            "matching_policy": (
                "One-to-one match PROBABLE_COUNTER_INTERACTION direction and timestamp to "
                "manually reviewed action interval; duplicate predictions are false positives. "
                "Other diagnostic markers reported separately."
            ),
        },
    )
    print("Prepared source hashes and fixed diagnostic protocol.")


def owner_proxy(obj, people, config):
    point = np.array(obj["center"])
    distances = sorted(
        (
            float(
                np.linalg.norm(
                    point - [(p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2]
                )
            ),
            n,
        )
        for n, p in enumerate(people)
        if p.get("identity_reliable", True) and p["role"] != "unknown"
    )
    if not distances:
        return "unknown"
    d, idx = distances[0]
    margin = distances[1][0] - d if len(distances) > 1 else config["association_radius"]
    if d <= config["association_radius"] and margin >= config["association_margin"]:
        return people[idx]["role"]
    return "unknown"


def infer_missing():
    # Existing detector/trackers only; no EventDetector or RuleEngine invocation.
    import torch

    from app.cv.analyze_events import suppress_nested_people
    from app.cv.inspect_cash import CashDetector
    from app.cv.train_cash import local_yolo
    from app.events.config import EventConfig
    from app.events.observations import suppress_object_duplicates
    from app.events.tracking import LogicalObjectTracker
    from app.services.person_roles import is_reflection_candidate
    from app.services.role_tracking import GeometricRoleTracker

    assert torch.cuda.is_available(), "Use the installed RTX GPU for fresh inference"
    config = EventConfig.model_validate_json(
        (ROOT / "config/event_camera.json").read_text(encoding="utf-8")
    )
    yolo = local_yolo()
    person_model = yolo(str(ROOT / "yolo11n.pt"))
    cash = CashDetector(
        ROOT / "data/cash/models/cash.pt", config.minimum_detection_confidence, device=0
    )
    for source in read_json(META / "sources.json"):
        out = ROOT / source["cache"]
        if (out / "detections.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(source["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, round(fps / config.sample_fps))
        roles = GeometricRoleTracker(max_gap_sec=config.person_track_gap_sec)
        objects = LogicalObjectTracker(config)
        raws, states, observations = [], [], []
        index = 0
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            t = index / fps
            h, w = frame.shape[:2]
            pred = person_model.predict(
                frame,
                classes=[0],
                conf=config.minimum_detection_confidence,
                imgsz=640,
                device=0,
                verbose=False,
            )[0]
            raw_people = [
                {"box_xyxy": b, "confidence": c}
                for b, c in zip(
                    pred.boxes.xyxy.cpu().tolist(), pred.boxes.conf.cpu().tolist(), strict=True
                )
            ]
            raw_objects = [
                {
                    "label": "cash",
                    "confidence": d["confidence"],
                    "bbox": [v / (w if j % 2 == 0 else h) for j, v in enumerate(d["box_xyxy"])],
                }
                for d in cash.detect(frame)
            ]
            people = [
                {
                    **p,
                    "counted": not is_reflection_candidate(
                        p["box_xyxy"], w, h, config.reflection_regions
                    ),
                }
                for p in raw_people
            ]
            people, _ = suppress_nested_people(
                [p for p in people if p["counted"]], config.person_duplicate_overlap
            )
            roles.update(people, t, w, h, config.zones, config.cashier_anchor)
            normalized = [
                {
                    "track_id": p["track_id"],
                    "role": p["role"],
                    "confidence": p["confidence"],
                    "identity_reliable": not p.get("tracking_ambiguous", False),
                    "bbox": [v / (w if j % 2 == 0 else h) for j, v in enumerate(p["box_xyxy"])],
                }
                for p in people
            ]
            kept, suppressed = suppress_object_duplicates(raw_objects, config.object_duplicate_iou)
            tracked = objects.update(kept, t)
            raws.append(
                {
                    "frame_index": index,
                    "timestamp_sec": t,
                    "width": w,
                    "height": h,
                    "people": raw_people,
                    "objects": raw_objects,
                }
            )
            state = {
                "timestamp_sec": t,
                "people": normalized,
                "objects": tracked,
                "suppressed_object_detections": suppressed,
            }
            states.append(state)
            observations.extend(tracked)
            debug = frame.copy()
            polygon = np.rint(np.array(config.zones["counter"]) * [w, h]).astype(np.int32)
            cv2.polylines(debug, [polygon], True, (0, 255, 255), 1)
            for p in normalized + tracked:
                b = [round(v * (w if j % 2 == 0 else h)) for j, v in enumerate(p["bbox"])]
                cv2.rectangle(debug, tuple(b[:2]), tuple(b[2:]), (0, 220, 60), 1)
                cv2.putText(
                    debug,
                    p.get("role", "cash") + " " + p["track_id"],
                    (b[0], max(12, b[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 220, 60),
                    1,
                )
            cv2.putText(
                debug,
                f"DIAGNOSTIC INPUT ONLY {t:.2f}s",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h)
                )
                assert writer.isOpened()
            writer.write(debug)
            if len(raws) % 60 == 0:
                print(f"{source['id']}: {t:.1f}s, GPU inference only", flush=True)
            index += 1
        cap.release()
        if writer:
            writer.release()
        write_json(
            out / "detections.json",
            {
                "video_sha256": source["video_sha256"],
                "fps": fps,
                "stride": stride,
                "person_weights_sha256": sha(ROOT / "yolo11n.pt"),
                "cash_weights_sha256": cash.model_sha256,
                "device": torch.cuda.get_device_name(),
                "frames": raws,
            },
        )
        write_json(out / "track_history.json", {"frames": states, "observations": observations})


def features():
    config = read_json(ROOT / "config/event_camera.json")
    protocol = read_json(META / "protocol.json")
    for source in read_json(META / "sources.json"):
        cache = ROOT / source["cache"]
        raw = read_json(cache / "detections.json")
        assert raw["video_sha256"] == source["video_sha256"]
        assert raw["cash_weights_sha256"] == sha(ROOT / "data/cash/models/cash.pt")
        states = read_json(cache / "track_history.json")["frames"]
        assert len(states) == len(raw["frames"])
        cap = cv2.VideoCapture(source["video"])
        raw_by_idx = {r["frame_index"]: (r, s) for r, s in zip(raw["frames"], states, strict=True)}
        index, previous, output = 0, None, []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index not in raw_by_idx:
                index += 1
                continue
            r, state = raw_by_idx[index]
            h, w = image.shape[:2]
            poly = np.array(config["zones"]["counter"], np.float32) * [w, h]
            px, py, pw, ph = cv2.boundingRect(np.rint(poly).astype(np.int32))
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 255)
            gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(
                np.float32
            )
            motion = np.zeros_like(mask)
            if previous is not None:
                delta = gray - previous
                delta -= np.median(delta[mask > 0])
                motion = ((np.abs(delta) >= protocol["motion_gray_delta"]) & (mask > 0)).astype(
                    np.uint8
                )
                motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fraction = float(motion.sum() / np.count_nonzero(mask))
            previous = gray
            people = state["people"]
            cashier = [
                p for p in people if p["role"] == "cashier" and p.get("identity_reliable", True)
            ]

            def box_gap(person, w=w, h=h, px=px, py=py, pw=pw, ph=ph):
                a, b, c, d = np.array(person["bbox"]) * [w, h, w, h]
                return float(
                    np.hypot(max(0, a - (px + pw), px - c), max(0, b - (py + ph), py - d)) / pw
                )

            gap = min((box_gap(p) for p in cashier), default=1e6)
            other_gap = min((box_gap(p) for p in people if p["role"] != "cashier"), default=1e6)
            objects = []
            for original in state["objects"]:
                obj = copy.deepcopy(original)
                point = np.array(obj["center"]) * [w, h]
                signed = cv2.pointPolygonTest(
                    poly.astype(np.float32), tuple(map(float, point)), True
                )
                obj["inside_counter"] = signed >= 0
                obj["counter_distance_widths"] = max(0.0, -signed) / pw
                obj["proxy_owner"] = owner_proxy(obj, people, config)
                obj["center_counter_units"] = (point / pw).tolist()
                objects.append(obj)
            motion_evidence = bool(
                fraction >= protocol["motion_fraction"] and gap <= protocol["reach_box_gap_widths"]
            )
            out = {
                "frame_index": index,
                "timestamp_sec": r["timestamp_sec"],
                "width": w,
                "height": h,
                "counter_width_px": pw,
                "people": people,
                "objects": objects,
                "counter_motion_fraction": fraction,
                "cashier_visible": bool(cashier),
                "cashier_box_gap_widths": gap,
                "other_person_box_gap_widths": other_gap,
                "cashier_reach_motion_proxy": motion_evidence,
                "actor_ambiguous": other_gap <= protocol["reach_box_gap_widths"],
            }
            output.append(out)
            index += 1
        cap.release()
        write_json(
            DATA / source["id"] / "features.json",
            {
                "source_video_sha256": source["video_sha256"],
                "detections_sha256": sha(cache / "detections.json"),
                "tracks_sha256": sha(cache / "track_history.json"),
                "frames": output,
                "duration_sec": index / raw["fps"],
                "fps": raw["fps"],
                "stride": raw["stride"],
            },
        )
        print(
            source["id"],
            len(output),
            "frames, reach/motion proxy:",
            [round(f["timestamp_sec"], 2) for f in output if f["cashier_reach_motion_proxy"]],
        )


def review():
    # Human review sampling metadata only, never loaded by the inference module.
    times = {
        "x3": [15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19.5],
        "x2": [12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16.5],
        "x1": [13.5, 14, 14.5, 15, 16, 18.5, 19, 19.5, 20, 20.5, 22, 24],
        "x5": [
            0,
            8,
            12,
            19,
            21,
            24,
            25,
            26,
            27,
            28,
            30,
            31,
            31.5,
            32,
            32.5,
            33,
            34,
            45,
            60,
            65,
            69,
            71,
        ],
    }
    for source in read_json(META / "sources.json"):
        cap = cv2.VideoCapture(source["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        debug = cv2.VideoCapture(str(ROOT / source["cache"] / "debug.mp4"))
        dfps = debug.get(cv2.CAP_PROP_FPS)
        panels = []
        for t in times[source["id"]]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            debug.set(cv2.CAP_PROP_POS_FRAMES, round(t * dfps))
            dok, dframe = debug.read()
            # Wider image includes cashier forearm, counter and customer handover region.
            crop = Image.fromarray(cv2.cvtColor(frame[350:590, 15:290], cv2.COLOR_BGR2RGB)).resize(
                (412, 360)
            )
            panel = Image.new("RGB", (610, 390), "white")
            panel.paste(crop, (0, 25))
            if dok:
                thumb = Image.fromarray(cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB))
                thumb.thumbnail((195, 350))
                panel.paste(thumb, (415, 25))
            ImageDraw.Draw(panel).text(
                (5, 5), f"{source['id']} {t:.2f}s RAW + EXISTING DEBUG", fill="black"
            )
            panels.append(panel)
        cap.release()
        debug.release()
        for start in range(0, len(panels), 8):
            page = Image.new(
                "RGB", (1220, 390 * ((min(8, len(panels) - start) + 1) // 2)), "#eeeeee"
            )
            for k, panel in enumerate(panels[start : start + 8]):
                page.paste(panel, ((k % 2) * 610, (k // 2) * 390))
            page.save(DATA / f"{source['id']}-review-{start // 8}.jpg", quality=92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "infer-missing", "features", "review"])
    args = parser.parse_args()
    {"prepare": prepare, "infer-missing": infer_missing, "features": features, "review": review}[
        args.command
    ]()
