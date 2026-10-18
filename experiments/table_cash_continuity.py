"""Independent pixel-continuity audit. No production imports or semantic event emission."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "datasets/table_cash_continuity_v1"
DATA = ROOT / "data/table_cash_continuity_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_image(path, image):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise ValueError("Image encoding failed")
    encoded.tofile(str(path))


def box_center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = np.prod(np.array(a[2:]) - a[:2]) + np.prod(np.array(b[2:]) - b[:2]) - intersection
    return float(intersection / max(union, 1e-6))


def inside(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.float32), tuple(point), False) >= 0


def prepare():
    META.mkdir(exist_ok=True)
    if (META / "protocol.json").exists():
        raise ValueError("Protocol already frozen")
    write(
        META / "protocol.json",
        {
            "candidate_max_gap_sec": 5.0,
            "candidate_max_center_distance_normalized": 0.18,
            "minimum_track_observations": 3,
            "flow_min_points": 5,
            "flow_min_survival": 0.5,
            "flow_max_fb_error_px": 1.5,
            "flow_max_endpoint_error_box_diagonals": 0.5,
            "flow_min_endpoint_iou": 0.3,
            "template_min_ncc": 0.8,
            "template_min_std": 8,
            "appearance_min_similarity": 0.7,
            "fixed_roi_min_match_fraction": 0.9,
            "maximum_fixed_roi_unknown_sec": 0.25,
            "wrist_guard_px": 25,
            "sample_review_interval_sec": 0.2,
            "notes": [
                "Exploratory thresholds frozen before measurement; no desired-outcome calibration.",
                "Similarity scores/confidence are heuristic, not calibrated probabilities.",
                "Fixed-ROI match is appearance persistence, not proof of cash or identity.",
                "Native video FPS for optical flow; no reseeding during gaps.",
                "All table-related fragment pairs within the diagnostic horizon are inspected.",
            ],
        },
    )
    sources = read(ROOT / "datasets/service_boundaries_v1/sources.json")
    for source in sources:
        source["tracks"] = f"data/service_boundaries_v1/{source['id']}/legacy/track_history.json"
        source["events"] = f"data/service_boundaries_v1/{source['id']}/legacy/events.json"
    write(META / "sources.json", sources)
    paths = [
        p
        for folder in ("app", "config")
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    paths += [ROOT / "data/cash/models/cash.pt", ROOT / "yolo11n.pt"]
    write(META / "protected-sha256.json", {str(p.relative_to(ROOT)): sha(p) for p in paths})


def candidates(tracks, zones, protocol):
    pairs = []
    for src, a in tracks.items():
        for dst, b in tracks.items():
            gap = b[0]["timestamp_sec"] - a[-1]["timestamp_sec"]
            if src == dst or not 0 < gap <= protocol["candidate_max_gap_sec"]:
                continue
            if not inside(a[-1]["center"], zones["table"]) or not inside(
                b[0]["center"], zones["table"]
            ):
                continue
            if (
                math.dist(a[-1]["center"], b[0]["center"])
                > protocol["candidate_max_center_distance_normalized"]
            ):
                continue
            pairs.append(
                {
                    "source_track": src,
                    "target_track": dst,
                    "start_sec": a[-1]["timestamp_sec"],
                    "end_sec": b[0]["timestamp_sec"],
                    "time_gap": gap,
                    "source_observations": len(a),
                    "target_observations": len(b),
                }
            )
    return sorted(pairs, key=lambda p: (p["start_sec"], p["end_sec"]))


class VideoROI:
    def __init__(self, path, zone):
        cap = cv2.VideoCapture(str(path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width, self.height = int(cap.get(3)), int(cap.get(4))
        points = np.array(zone) * [self.width, self.height]
        x1, y1 = np.maximum(0, points.min(0) - 40).astype(int)
        x2, y2 = np.minimum([self.width, self.height], points.max(0) + 40).astype(int)
        self.bounds = [x1, y1, x2, y2]
        self.frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.frames.append(frame[y1:y2, x1:x2].copy())
        cap.release()
        self.grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frames]

    def index(self, t):
        return min(len(self.frames) - 1, round(t * self.fps))

    def box(self, observation):
        return (
            np.array(observation["bbox"]) * [self.width, self.height, self.width, self.height]
            - np.array(self.bounds[:2] * 2)
        ).tolist()


def crop(image, box):
    x1, y1, x2, y2 = np.rint(box).astype(int)
    h, w = image.shape[:2]
    return image[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]


def appearance(a, b):
    if min(a.shape[:2] + b.shape[:2]) < 3:
        return {"similarity": 0, "reason": "crop_too_small"}
    a, b = cv2.resize(a, (32, 24)), cv2.resize(b, (32, 24))

    def hist(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        h = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        return cv2.normalize(h, h, 1, 0, cv2.NORM_L1)

    color = 1 - cv2.compareHist(hist(a), hist(b), cv2.HISTCMP_BHATTACHARYYA)
    ga, gb = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(float) for x in (a, b)]
    ncc = float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1]) if min(ga.std(), gb.std()) > 1 else 0
    edge_a, edge_b = [cv2.Canny(x.astype(np.uint8), 40, 100) > 0 for x in (ga, gb)]
    edge_iou = float((edge_a & edge_b).sum() / max(1, (edge_a | edge_b).sum()))
    return {
        "similarity": round(float(0.5 * color + 0.5 * max(0, ncc)), 4),
        "lab_histogram_similarity": round(float(color), 4),
        "normalized_correlation": round(ncc, 4),
        "edge_iou": round(edge_iou, 4),
        "minimum_texture_std": round(float(min(ga.std(), gb.std())), 3),
    }


def optical_flow(video, first, last, box, protocol):
    gray = video.grays[first]
    mask = np.zeros_like(gray)
    x1, y1, x2, y2 = np.rint(box).astype(int)
    mask[
        max(0, y1 + 1) : min(gray.shape[0], y2 - 1), max(0, x1 + 1) : min(gray.shape[1], x2 - 1)
    ] = 255
    points = cv2.goodFeaturesToTrack(
        gray, maxCorners=40, qualityLevel=0.01, minDistance=2, mask=mask, blockSize=3
    )
    if points is None or len(points) < protocol["flow_min_points"]:
        return {
            "valid": False,
            "reason": "insufficient_object_texture",
            "initial_points": 0 if points is None else len(points),
            "path": [],
        }
    original = points.copy()
    initial_count = len(points)
    path, direction = [], 1 if last >= first else -1
    for index in range(first + direction, last + direction, direction):
        target = video.grays[index]
        new, st, _ = cv2.calcOpticalFlowPyrLK(
            gray, target, points, None, winSize=(11, 11), maxLevel=2
        )
        if new is None:
            break
        back, sb, _ = cv2.calcOpticalFlowPyrLK(
            target, gray, new, None, winSize=(11, 11), maxLevel=2
        )
        good = (
            (st[:, 0] == 1)
            & (sb[:, 0] == 1)
            & (
                np.linalg.norm(points[:, 0] - back[:, 0], axis=1)
                <= protocol["flow_max_fb_error_px"]
            )
        )
        points, original = new[good], original[good]
        if (
            len(points) < protocol["flow_min_points"]
            or len(points) / initial_count < protocol["flow_min_survival"]
        ):
            path.append(
                {
                    "frame_index": index,
                    "timestamp_sec": index / video.fps,
                    "surviving_points": len(points),
                    "valid": False,
                }
            )
            break
        shifts = points[:, 0] - original[:, 0]
        shift = np.median(shifts, axis=0)
        scatter = float(np.median(np.linalg.norm(shifts - shift, axis=1)))
        estimated = (np.array(box) + [*shift, *shift]).tolist()
        path.append(
            {
                "frame_index": index,
                "timestamp_sec": index / video.fps,
                "bbox": estimated,
                "surviving_points": len(points),
                "valid": True,
                "motion_scatter_px": scatter,
            }
        )
        gray = target
    reached = bool(path and path[-1]["frame_index"] == last and path[-1]["valid"])
    return {
        "valid": reached,
        "reason": "tracked_all_frames" if reached else "points_lost_or_occluded",
        "initial_points": initial_count,
        "final_points": len(points),
        "survival": len(points) / initial_count,
        "path": path,
        "endpoint_box": path[-1].get("bbox") if reached else None,
    }


def fixed_roi(video, indices, seed, box, protocol):
    template = crop(video.grays[seed], box)
    textured = template.std() >= protocol["template_min_std"]
    observations = []
    for i in indices:
        patch = crop(video.grays[i], box)
        if patch.shape != template.shape:
            raise ValueError("ROI geometry changed")
        ncc = (
            float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            if textured and patch.std() > 1
            else 0
        )
        delta = float(np.mean(np.abs(patch.astype(float) - template.astype(float))))
        observations.append(
            {
                "timestamp_sec": i / video.fps,
                "frame_index": i,
                "ncc": ncc,
                "mean_absolute_difference": delta,
                "appearance_present": bool(textured and ncc >= protocol["template_min_ncc"]),
            }
        )
    longest, run = 0, 0
    for obs in observations:
        run = 0 if obs["appearance_present"] else run + 1
        longest = max(longest, run)
    return {
        "template_texture_std": float(template.std()),
        "valid_template": bool(textured),
        "match_fraction": mean_bool([o["appearance_present"] for o in observations]),
        "longest_unmatched_sec": longest / video.fps,
        "observations": observations,
    }


def mean_bool(values):
    return sum(values) / max(1, len(values))


def wrist_evidence(hands, t, box, offset, guard):
    nearest = min(hands, key=lambda f: abs(f["timestamp_sec"] - t))
    if abs(nearest["timestamp_sec"] - t) > 0.15:
        return {"status": "unknown", "reason": "wrist_observation_missing"}
    distances = []
    for hand in nearest["hands"]:
        if hand["owner"] != "cashier" or hand["wrist_confidence"] < 0.5:
            continue
        xy = np.array(hand["xy"]) - offset
        distances.append(float(np.linalg.norm(xy - np.clip(xy, box[:2], box[2:]))))
    if not distances:
        return {"status": "unknown", "reason": "cashier_wrist_unobserved"}
    return {
        "status": "near" if min(distances) <= guard else "clear_proxy",
        "minimum_wrist_distance_px": min(distances),
    }


def flow_metrics(flow, target, video, template, protocol):
    if not flow["valid"]:
        return {"endpoint_agrees": False, "appearance_p10": 0}
    diagonal = max(1.0, float(np.linalg.norm(np.array(target[2:]) - target[:2])))
    error = float(np.linalg.norm(box_center(flow["endpoint_box"]) - box_center(target)))
    iou = box_iou(flow["endpoint_box"], target)
    values = [
        appearance(template, crop(video.frames[p["frame_index"]], p["bbox"]))["similarity"]
        for p in flow["path"][::3]
        if p.get("bbox")
    ]
    return {
        "endpoint_agrees": error / diagonal <= protocol["flow_max_endpoint_error_box_diagonals"]
        and iou >= protocol["flow_min_endpoint_iou"],
        "endpoint_error_px": error,
        "endpoint_iou": iou,
        "appearance_p10": float(np.percentile(values, 10)) if values else 0,
    }


def measure_pair(pair, tracks, video, hands, protocol, out):
    source, target = tracks[pair["source_track"]], tracks[pair["target_track"]]
    a, b = source[-1], target[0]
    first, last = video.index(pair["start_sec"]), video.index(pair["end_sec"])
    ba, bb = video.box(a), video.box(b)
    ca, cb = crop(video.frames[first], ba), crop(video.frames[last], bb)
    compared = [
        appearance(
            crop(video.frames[video.index(x["timestamp_sec"])], video.box(x)),
            crop(video.frames[video.index(y["timestamp_sec"])], video.box(y)),
        )
        for x in source[-3:]
        for y in target[:3]
    ]
    endpoint_appearance = appearance(ca, cb)
    aw, ah = np.array(ba[2:]) - ba[:2]
    bw, bh = np.array(bb[2:]) - bb[:2]
    displacement = float(np.linalg.norm(box_center(ba) - box_center(bb)))
    spatial = {
        "source_bbox_px": (np.array(ba) + video.bounds[:2] * 2).tolist(),
        "target_bbox_px": (np.array(bb) + video.bounds[:2] * 2).tolist(),
        "center_displacement_px": displacement,
        "endpoint_iou": box_iou(ba, bb),
        "source_size_px": [aw, ah],
        "target_size_px": [bw, bh],
        "visible_area_ratio": float(max(aw * ah / (bw * bh), bw * bh / (aw * ah))),
        "similarity": float(math.exp(-displacement / max(1, math.hypot(aw, ah)))),
        "same_fixed_roi": box_iou(ba, bb) >= 0.3
        and displacement <= 0.5 * max(math.hypot(aw, ah), math.hypot(bw, bh)),
    }
    forward = optical_flow(video, first, last, ba, protocol)
    backward = optical_flow(video, last, first, bb, protocol)
    forward.update(flow_metrics(forward, bb, video, ca, protocol))
    backward.update(flow_metrics(backward, ba, video, cb, protocol))
    source_roi = fixed_roi(video, range(first, last + 1), first, ba, protocol)
    target_roi = fixed_roi(video, range(first, last + 1), last, bb, protocol)
    wrist = [
        wrist_evidence(hands, i / video.fps, ba, video.bounds[:2], protocol["wrist_guard_px"])
        for i in range(first, last + 1)
    ]
    reasons = []
    temporal_support = min(len(source), len(target)) >= protocol["minimum_track_observations"]
    if not temporal_support:
        reasons.append("one_or_both_track_fragments_have_fewer_than_three_observations")
    if not spatial["same_fixed_roi"]:
        reasons.append("different_fixed_table_positions")
    if min(aw, ah, bw, bh) < 16:
        reasons.append("tiny_object_crop_limits_identity_detail")
    if endpoint_appearance["similarity"] < protocol["appearance_min_similarity"]:
        reasons.append("endpoint_appearance_not_consistent")
    if not forward["endpoint_agrees"] or not backward["endpoint_agrees"]:
        reasons.append("bidirectional_visual_path_not_established")
    if (
        min(forward["appearance_p10"], backward["appearance_p10"])
        < protocol["appearance_min_similarity"]
    ):
        reasons.append("object_appearance_not_preserved_along_path")
    if any(x["status"] == "near" for x in wrist):
        reasons.append("cashier_hand_near_object_possible_occlusion_or_hand_tracking")
    if any(x["status"] == "unknown" for x in wrist):
        reasons.append("wrist_visibility_incomplete")
    stationary_supported = (
        spatial["same_fixed_roi"]
        and source_roi["valid_template"]
        and source_roi["match_fraction"] >= protocol["fixed_roi_min_match_fraction"]
        and source_roi["longest_unmatched_sec"] <= protocol["maximum_fixed_roi_unknown_sec"]
    )
    motion_supported = (
        forward["endpoint_agrees"]
        and backward["endpoint_agrees"]
        and min(forward["appearance_p10"], backward["appearance_p10"])
        >= protocol["appearance_min_similarity"]
    )
    supported = (
        temporal_support
        and (stationary_supported or motion_supported)
        and endpoint_appearance["similarity"] >= protocol["appearance_min_similarity"]
        and not any(x["status"] != "clear_proxy" for x in wrist)
    )
    confidence = (
        0.25 * spatial["similarity"]
        + 0.25 * endpoint_appearance["similarity"]
        + 0.25 * float(motion_supported)
        + 0.25 * float(stationary_supported)
    )
    selected = sorted(
        set(
            [first, last]
            + [round(x) for x in np.linspace(first, last, 5)]
            + [f["path"][-1]["frame_index"] for f in (forward, backward) if f["path"]]
        )
    )
    sheet = out / "evidence" / f"{pair['source_track']}--{pair['target_track']}.jpg"
    contact_sheet(
        video,
        selected,
        [("source", ba, (0, 255, 255)), ("target", bb, (255, 255, 0))],
        sheet,
        f"{pair['source_track']} -> {pair['target_track']}",
    )
    return {
        **pair,
        "spatial_similarity": spatial,
        "appearance_similarity": endpoint_appearance,
        "multi_observation_appearance": {
            "median": float(np.median([x["similarity"] for x in compared])),
            "maximum": max(x["similarity"] for x in compared),
        },
        "forward_flow": forward,
        "backward_flow": backward,
        "source_fixed_roi": source_roi,
        "target_fixed_roi": target_roi,
        "wrist_status_counts": dict(Counter(x["status"] for x in wrist)),
        "stationary_presence_supported": bool(stationary_supported),
        "motion_continuity_supported": bool(motion_supported),
        "status": "visual_continuity_supported" if supported else "insufficient_evidence",
        "identity_status": "insufficient_evidence",
        "confidence": round(confidence, 4),
        "confidence_semantics": "heuristic diagnostic support; not probability of same banknotes",
        "uncertainty": reasons + ["appearance_is_not_currency_identity_or_transaction_provenance"],
        "evidence_frames": [{"frame_index": i, "timestamp_sec": i / video.fps} for i in selected],
        "evidence_contact_sheet": str(sheet.relative_to(ROOT)),
    }


def measure_boundaries(tracks, video, sessions, hands, protocol, out):
    results = []
    for session in sessions[1:]:
        t = session["start_sec"]
        for key, nodes in tracks.items():
            past = [n for n in nodes if n["timestamp_sec"] < t]
            if len(past) < protocol["minimum_track_observations"]:
                continue
            seed = past[-1]
            first, last = video.index(seed["timestamp_sec"]), video.index(t)
            box = video.box(seed)
            observed = fixed_roi(
                video, range(first, min(len(video.frames), last + 4)), first, box, protocol
            )
            at_boundary = next(o for o in observed["observations"] if o["frame_index"] == last)
            guard = wrist_evidence(hands, t, box, video.bounds[:2], protocol["wrist_guard_px"])
            support = at_boundary["appearance_present"] and guard["status"] == "clear_proxy"
            image_path = out / "boundary-evidence" / f"{key}--{session['session_id']}.jpg"
            contact_sheet(
                video,
                [first, last],
                [("fixed ROI", box, (0, 255, 255))],
                image_path,
                f"{key}; {session['session_id']}",
            )
            results.append(
                {
                    "track": key,
                    "next_session_id": session["session_id"],
                    "boundary_sec": t,
                    "seed_sec": seed["timestamp_sec"],
                    "age_sec": t - seed["timestamp_sec"],
                    "appearance_at_boundary": at_boundary,
                    "wrist": guard,
                    "appearance_presence_supported": bool(support),
                    "identity_status": "insufficient_evidence",
                    "reason": "Fixed-location appearance does not prove previous-operation cash.",
                    "fixed_roi": observed,
                    "evidence_contact_sheet": str(image_path.relative_to(ROOT)),
                }
            )
    return results


def run():
    protocol = read(META / "protocol.json")
    zones = read(ROOT / "config/cash_security_camera.json")["zones"]
    summary = []
    for source in read(META / "sources.json"):
        if sha(source["video"]) != source["video_sha256"]:
            raise ValueError("Video hash mismatch")
        doc = read(ROOT / source["events"])
        if doc["video_sha256"] != source["video_sha256"]:
            raise ValueError("Track cache belongs to another video")
        tracks = defaultdict(list)
        for frame in read(ROOT / source["tracks"])["frames"]:
            for obj in frame["objects"]:
                if obj["label"] == "cash":
                    tracks[obj["logical_object_id"]].append(obj)
        pairs = candidates(tracks, zones, protocol)
        video = VideoROI(source["video"], zones["table"])
        hands = read(ROOT / source["auxiliary"] / "hands.json")["frames"]
        out = DATA / source["id"]
        measured = []
        for pair in pairs:
            measured.append(measure_pair(pair, tracks, video, hands, protocol, out))
        boundary = measure_boundaries(
            tracks, video, doc["rule_result"]["sessions"], hands, protocol, out
        )
        write(out / "pairs.json", measured)
        write(out / "boundaries.json", boundary)
        totals = {
            "video": source["id"],
            "candidate_pairs": len(measured),
            "visual_continuity_supported": sum(
                p["status"] == "visual_continuity_supported" for p in measured
            ),
            "identity_confirmed": 0,
            "r05_events_created": 0,
            "boundary_appearance_supported": [
                b["track"] for b in boundary if b["appearance_presence_supported"]
            ],
            "sessions": doc["rule_result"]["sessions"],
        }
        summary.append(totals)
        print(totals, flush=True)
        del video
    write(DATA / "summary.json", summary)
    protected = read(META / "protected-sha256.json")
    checks = {p: sha(ROOT / p) == v for p, v in protected.items()}
    write(DATA / "protected-check.json", checks)
    if not all(checks.values()):
        raise ValueError("Protected production files changed")
    report()


def report():
    # Keep coordinate systems and the pre-existing provenance explicit in every audit record.
    zones = read(ROOT / "config/cash_security_camera.json")["zones"]
    for source in read(META / "sources.json"):
        cap = cv2.VideoCapture(source["video"])
        width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        points = np.array(zones["table"]) * [width, height]
        bounds = (
            np.concatenate(
                (np.maximum(0, points.min(0) - 40), np.minimum([width, height], points.max(0) + 40))
            )
            .astype(int)
            .tolist()
        )
        doc = read(ROOT / source["events"])
        provenance = doc["cash_provenance"]
        groups = {g["logical_cash_id"]: g for g in provenance["objects"]}
        path = DATA / source["id"] / "pairs.json"
        pairs = read(path)
        for pair in pairs:
            pair["flow_coordinate_system"] = {
                "units": "ROI pixels",
                "roi_bounds_xyxy": bounds,
                "video_size": [width, height],
                "fps": fps,
            }
            group = groups[provenance["aliases"][pair["source_track"]]]
            pair["source_provenance_before_experiment"] = {
                "source_session_id": group["source_session_id"],
                "status": group["status"],
                "provenance_confidence": group["provenance_confidence"],
                "uncertainty": group["uncertainty"],
            }
            pair["target_origin_after_experiment"] = "insufficient_evidence; not reassigned"
        write(path, pairs)
    body = "<h1>Table cash continuity: independent diagnostic</h1>"
    body += (
        "<p>Similarity and optical flow are supporting observations. No identities merged; "
        "no semantic events emitted. Production unchanged.</p>"
    )
    if (META / "review.json").exists():
        review = read(META / "review.json")
        body += "<h2>Выводы визуальной проверки</h2>"
        body += "<p>" + html.escape(review["conclusion"]) + "</p>"
        for item in review["items"]:
            body += "<h3>" + html.escape(item["video"] + ": " + item["subject"]) + "</h3>"
            body += "<p>" + html.escape(item["finding"]) + "</p>"
            body += "<p>Статус: " + html.escape(item["status"]) + "</p>"
            for evidence in item["evidence"]:
                relative = Path(evidence).relative_to(DATA.relative_to(ROOT)).as_posix()
                body += f'<p><a href="{relative}">Кадры доказательств</a></p>'
    if (DATA / "control-summary.json").exists():
        body += "<h2>Контрольные пропуски YOLO на неизменённых реальных кадрах</h2><pre>"
        body += html.escape(json.dumps(read(DATA / "control-summary.json"), indent=2)) + "</pre>"
    for row in read(DATA / "summary.json"):
        name = row["video"]
        body += f"<h2>{html.escape(name)}</h2><pre>{html.escape(json.dumps(row, indent=2))}</pre>"
        body += (
            "<table><tr><th>Pair</th><th>Gap, s</th><th>Distance, px</th><th>Appearance</th>"
            "<th>Flow end IoU F/B</th><th>Fixed ROI match</th><th>Support</th></tr>"
        )
        for p in read(DATA / name / "pairs.json"):
            image_path = (
                Path(p["evidence_contact_sheet"]).relative_to(DATA.relative_to(ROOT)).as_posix()
            )
            body += (
                f'<tr><td><a href="{image_path}">{p["source_track"]} → {p["target_track"]}</a></td>'
                f"<td>{p['time_gap']:.3f}</td><td>{p['spatial_similarity']['center_displacement_px']:.1f}</td>"
                f"<td>{p['appearance_similarity']['similarity']:.3f}</td>"
                f"<td>{p['forward_flow'].get('endpoint_iou', 0):.3f} / "
                f"{p['backward_flow'].get('endpoint_iou', 0):.3f}</td>"
                f"<td>{p['source_fixed_roi']['match_fraction']:.1%}</td><td>{p['status']}</td></tr>"
            )
        body += "</table>"
        body += (
            f'<p><a href="{name}/pairs.json">All metrics, evidence and doubts</a> · '
            f'<a href="{name}/boundaries.json">Fixed regions at next service</a></p>'
        )
    (DATA / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:15px sans-serif;margin:25px}'
        "td,th{padding:6px;border:1px solid #ccc}"
        "table{border-collapse:collapse}pre{white-space:pre-wrap}</style>" + body,
        encoding="utf-8",
    )


def controls():
    """Measured-image controls: hide intermediate YOLO samples, never alter pixels/events."""
    path = META / "controls-protocol.json"
    if not path.exists():
        write(
            path,
            {
                "window_observations": 9,
                "minimum_confidence": 0.7,
                "maximum_center_spread_normalized": 0.008,
                "notes": (
                    "Post-diagnostic sensitivity check on stable tracks; "
                    "not independent validation or threshold tuning."
                ),
            },
        )
    control = read(path)
    protocol = read(META / "protocol.json")
    zones = read(ROOT / "config/cash_security_camera.json")["zones"]
    totals = []
    for source in read(META / "sources.json"):
        tracks = defaultdict(list)
        for f in read(ROOT / source["tracks"])["frames"]:
            for o in f["objects"]:
                if o["label"] == "cash":
                    tracks[o["logical_object_id"]].append(o)
        video = VideoROI(source["video"], zones["table"])
        hands = read(ROOT / source["auxiliary"] / "hands.json")["frames"]
        output = []
        for key, observations in tracks.items():
            next_allowed = -1
            for start in range(len(observations) - control["window_observations"] + 1):
                window = observations[start : start + control["window_observations"]]
                if window[0]["timestamp_sec"] < next_allowed:
                    continue
                if not all(
                    x["confidence"] >= control["minimum_confidence"]
                    and inside(x["center"], zones["table"])
                    for x in window
                ):
                    continue
                if (
                    max(math.dist(a["center"], b["center"]) for a in window for b in window)
                    > control["maximum_center_spread_normalized"]
                ):
                    continue
                a, b = window[:3], window[-3:]
                pair = {
                    "source_track": f"{key}-window-{start}-A",
                    "target_track": f"{key}-window-{start}-B",
                    "start_sec": a[-1]["timestamp_sec"],
                    "end_sec": b[0]["timestamp_sec"],
                    "time_gap": b[0]["timestamp_sec"] - a[-1]["timestamp_sec"],
                    "source_observations": 3,
                    "target_observations": 3,
                }
                measured = measure_pair(
                    pair,
                    {pair["source_track"]: a, pair["target_track"]: b},
                    video,
                    hands,
                    protocol,
                    DATA / source["id"] / "controls",
                )
                measured["control_type"] = (
                    "withheld_intermediate_YOLO_observations_real_unchanged_pixels"
                )
                measured["original_object_track"] = key
                output.append(measured)
                next_allowed = window[-1]["timestamp_sec"] + 0.01
        write(DATA / source["id"] / "controls.json", output)
        # Inspect backgrounds before a seed is first detected; high similarity is a warning,
        # not proof of empty table (YOLO might have missed a real object earlier).
        warnings = []
        for boundary in read(DATA / source["id"] / "boundaries.json"):
            if not boundary["appearance_presence_supported"]:
                continue
            nodes = tracks[boundary["track"]]
            seed = min(nodes, key=lambda n: abs(n["timestamp_sec"] - boundary["seed_sec"]))
            box = video.box(seed)
            indices = sorted(
                {
                    video.index(max(0, nodes[0]["timestamp_sec"] - offset))
                    for offset in (2, 1.5, 1, 0.5)
                }
            )
            reference = fixed_roi(video, indices, video.index(seed["timestamp_sec"]), box, protocol)
            warnings.append(
                {
                    "track": boundary["track"],
                    "pre_detection_appearance": reference,
                    "uncertainty": (
                        "Pre-existing appearance may be background or an earlier "
                        "missed object; requires review."
                    ),
                }
            )
        write(DATA / source["id"] / "background-controls.json", warnings)
        total = {
            "video": source["id"],
            "windows": len(output),
            "stationary_supported": sum(x["stationary_presence_supported"] for x in output),
            "full_visual_support": sum(
                x["status"] == "visual_continuity_supported" for x in output
            ),
        }
        totals.append(total)
        print(total, flush=True)
        del video
    write(DATA / "control-summary.json", totals)


def contact_sheet(video, indices, boxes, path, title):
    panels = []
    for index in sorted(set(indices)):
        image = video.frames[index].copy()
        for label, box, color in boxes:
            a = np.rint(box).astype(int)
            cv2.rectangle(image, tuple(a[:2]), tuple(a[2:]), color, 1)
            cv2.putText(
                image, label, (a[0], max(12, a[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
            )
        image = cv2.resize(
            image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_NEAREST
        )
        panel = Image.new("RGB", (image.shape[1], image.shape[0] + 38), "white")
        panel.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (0, 38))
        ImageDraw.Draw(panel).text(
            (5, 3), f"{title}\n{index / video.fps:.3f}s frame={index}", fill="black"
        )
        panels.append(panel)
    w, h = panels[0].size
    sheet = Image.new("RGB", (w * 3, h * math.ceil(len(panels) / 3)), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 3) * w, (i // 3) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def inspect_sources():
    protocol = read(META / "protocol.json")
    zones = read(ROOT / "config/cash_security_camera.json")["zones"]
    for source in read(META / "sources.json"):
        if sha(source["video"]) != source["video_sha256"]:
            raise ValueError("Video hash mismatch")
        frames = read(ROOT / source["tracks"])["frames"]
        tracks = defaultdict(list)
        for frame in frames:
            for obj in frame["objects"]:
                if obj["label"] == "cash":
                    tracks[obj["logical_object_id"]].append(obj)
        pairs = candidates(tracks, zones, protocol)
        out = DATA / source["id"]
        write(out / "candidates.json", pairs)
        video = VideoROI(source["video"], zones["table"])
        print(source["id"], len(pairs), "candidate pairs", flush=True)
        for pair in pairs:
            # Review all pairs with multi-observation support; singletons remain in machine audit.
            if (
                min(pair["source_observations"], pair["target_observations"])
                < protocol["minimum_track_observations"]
            ):
                continue
            a, b = tracks[pair["source_track"]][-1], tracks[pair["target_track"]][0]
            first, last = video.index(pair["start_sec"]), video.index(pair["end_sec"])
            indices = list(
                range(
                    max(0, first - 6),
                    min(len(video.frames) - 1, last + 6) + 1,
                    max(1, round(video.fps * protocol["sample_review_interval_sec"])),
                )
            ) + [first, last]
            contact_sheet(
                video,
                indices,
                [("source", video.box(a), (0, 255, 255)), ("target", video.box(b), (255, 255, 0))],
                out / "review" / f"{pair['source_track']}--{pair['target_track']}.jpg",
                f"{pair['source_track']} -> {pair['target_track']}",
            )
        del video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "inspect", "run", "report", "controls"])
    args = parser.parse_args()
    {
        "prepare": prepare,
        "inspect": inspect_sources,
        "run": run,
        "report": report,
        "controls": controls,
    }[args.command]()


if __name__ == "__main__":
    main()
