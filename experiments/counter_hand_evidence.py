"""Render original frames with wrist evidence; no production pipeline execution."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experiments.counter_hand_data import DATA, META
from experiments.counter_occupancy import ROOT, imwrite, read_json, write_json


def annotate(raw, hands, p, objects=()):
    image = raw.copy()
    h, w = image.shape[:2]
    for key, color in [
        ("counter_zone", (0, 200, 255)),
        ("counter_output_zone", (0, 255, 0)),
        ("output_core", (255, 255, 0)),
    ]:
        cv2.polylines(image, [np.rint(np.array(p[key]) * [w, h]).astype(np.int32)], True, color, 1)
    for hand in hands:
        if hand["xy"][1] < 0.38 * h:
            continue
        color = {"cashier": (255, 100, 0), "customer": (255, 0, 255), "unknown": (120, 120, 120)}[
            hand["owner"]
        ]
        wrist = tuple(np.rint(hand["xy"]).astype(int))
        shoulder = tuple(np.rint(hand["shoulder"][:2]).astype(int))
        elbow = tuple(np.rint(hand["elbow"][:2]).astype(int))
        if hand["arm_confidence"] >= p["joint_confidence"]:
            cv2.line(image, shoulder, elbow, color, 1)
            cv2.line(image, elbow, wrist, color, 1)
        cv2.circle(image, wrist, 3, color, -1)
        text = f"{hand['track_id']} {hand['owner'][:3]} {hand['wrist_confidence']:.2f}"
        cv2.putText(image, text, (wrist[0] + 3, wrist[1] - 3), 0, 0.28, color, 1, cv2.LINE_AA)
    for obj in objects:
        box = np.rint(np.array(obj["cash_bbox"]) * [w, h, w, h]).astype(int)
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 1)
        cv2.putText(
            image, f"cash {obj['cash_confidence']:.2f}", tuple(box[:2]), 0, 0.3, (0, 0, 255), 1
        )
    return image


def panel(raw, annotated, title):
    raw_crop = raw[int(raw.shape[0] * 0.43) : int(raw.shape[0] * 0.74), : int(raw.shape[1] * 0.68)]
    ann_crop = annotated[
        int(raw.shape[0] * 0.43) : int(raw.shape[0] * 0.74), : int(raw.shape[1] * 0.68)
    ]
    crops = [cv2.resize(c, (472, 393)) for c in [raw_crop, ann_crop]]
    canvas = Image.new("RGB", (944, 425), "white")
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black")
    for i, crop in enumerate(crops):
        canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (i * 472, 32))
    return canvas


def main():
    p = read_json(META / "protocol.json")
    for source in read_json(META / "sources.json"):
        out = DATA / source["id"]
        poses = read_json(out / "poses.json")
        frames = read_json(out / "hands.json")["frames"]
        prediction = read_json(out / "predictions.json")
        targets = set()
        for contact in prediction["contacts"]:
            targets.update(n["frame_index"] for n in contact["wrist_evidence"])
            targets.update(n["frame_index"] for n in contact["cash_links"])
        # Whole-video fixed sampling, not selected from known action times.
        review_indices = {
            round(n * poses["fps"] / poses["stride"]) * poses["stride"]
            for n in np.arange(0, poses["duration_sec"], 0.5)
        }
        targets |= review_indices
        by_index = {f["frame_index"]: f for f in frames}
        cap = cv2.VideoCapture(source["video"])
        writer = cv2.VideoWriter(
            str(out / "wrist-debug.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            poses["fps"] / poses["stride"],
            (464, 848),
        )
        assert writer.isOpened()
        index, panels, review_page = 0, [], 0
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            if index in by_index or index in targets:
                f = by_index.get(index) or min(frames, key=lambda n: abs(n["frame_index"] - index))
                cash = [
                    n
                    for c in prediction["contacts"]
                    for n in c["cash_links"]
                    if n["frame_index"] == index
                ]
                annotated = annotate(raw, f["hands"], p, cash)
                cv2.putText(
                    annotated,
                    f"{index / poses['fps']:.2f}s | diagnostic wrists",
                    (8, 24),
                    0,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                if index in by_index:
                    writer.write(annotated)
                if index in targets:
                    imwrite(out / "evidence" / f"{index:05d}-raw.jpg", raw)
                    imwrite(out / "evidence" / f"{index:05d}-wrists.jpg", annotated)
                if index in review_indices:
                    panels.append(
                        panel(
                            raw,
                            annotated,
                            f"{source['id']} {index / poses['fps']:.2f}s | raw / wrists",
                        )
                    )
                    if len(panels) == 12:
                        sheet = Image.new("RGB", (1888, 425 * 6), "white")
                        for i, im in enumerate(panels):
                            sheet.paste(im, ((i % 2) * 944, (i // 2) * 425))
                        sheet.save(out / f"review-{review_page}.jpg", quality=90)
                        panels, review_page = [], review_page + 1
            index += 1
        writer.release()
        cap.release()
        if panels:
            sheet = Image.new("RGB", (1888, 425 * ((len(panels) + 1) // 2)), "white")
            for i, im in enumerate(panels):
                sheet.paste(im, ((i % 2) * 944, (i // 2) * 425))
            sheet.save(out / f"review-{review_page}.jpg", quality=90)
        for contact in prediction["contacts"]:
            selected = (
                contact["wrist_evidence"] + contact["cash_links"][:3] + contact["cash_links"][-1:]
            )
            selected = list({n["frame_index"]: n for n in selected}.values())
            sheets = []
            contact["evidence_frames"] = []
            for node in selected:
                path = out / "evidence" / f"{node['frame_index']:05d}"
                raw = cv2.imdecode(np.fromfile(str(path) + "-raw.jpg", np.uint8), 1)
                annotated = cv2.imdecode(np.fromfile(str(path) + "-wrists.jpg", np.uint8), 1)
                sheets.append(
                    panel(
                        raw,
                        annotated,
                        f"{contact['id']} {node['timestamp_sec']:.3f}s "
                        f"| owner={contact['owner']} | raw / wrists",
                    )
                )
                contact["evidence_frames"].append(
                    {
                        "timestamp_sec": node["timestamp_sec"],
                        "raw": str(path.relative_to(ROOT)) + "-raw.jpg",
                        "diagnostic": str(path.relative_to(ROOT)) + "-wrists.jpg",
                    }
                )
            sheet = Image.new("RGB", (944, 425 * len(sheets)), "white")
            for i, im in enumerate(sheets):
                sheet.paste(im, (0, i * 425))
            sheet.save(out / f"{contact['id']}.jpg", quality=93)
        for marker in prediction["markers"]:
            marker["evidence_frames"] = [
                str(
                    (out / "evidence" / f"{marker['frame_index']:05d}-{kind}.jpg").relative_to(ROOT)
                )
                for kind in ["raw", "wrists"]
            ]
            contact = next(c for c in prediction["contacts"] if c["id"] == marker["contact_id"])
            marker["observed_features"] = contact["observed_features"]
        write_json(out / "predictions-with-evidence.json", prediction)
        print("Rendered", source["id"], flush=True)


if __name__ == "__main__":
    main()
