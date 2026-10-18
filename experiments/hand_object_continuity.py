"""Offline hand-region and pixel-continuity diagnostics; never business events.

An extrapolated palm is a geometric proxy, not a detected keypoint. Optical flow
tracks visible texture, not the identity or denomination of a banknote.
"""

from copy import deepcopy

import cv2
import numpy as np

PROTOCOL = {
    "version": 1,
    "palm_forearm_fraction": 0.5,
    "min_arm_confidence": 0.35,
    "flow_max_seconds": 1.0,
    "flow_min_points": 4,
    "flow_min_survival": 0.6,
    "flow_fb_error_px": 1.0,
    "flow_max_median_error": 20.0,
    "flow_min_template_correlation": 0.65,
    "flow_max_translation_residual_px": 2.0,
    "flow_min_seed_extent_fraction": 0.3,
    "flow_max_frame_gap_sec": 0.1,
    "confirmation": "hypotheses only; no production integration",
}


def palm_frames(frames, protocol=PROTOCOL):
    """Preserve roles/track IDs and original wrist; change only the position proxy."""
    result = deepcopy(frames)
    for frame in result:
        accepted = []
        for hand in frame["hands"]:
            wrist = np.asarray(hand["xy"], dtype=float)
            elbow = np.asarray(hand["elbow"][:2], dtype=float)
            if (
                hand["arm_confidence"] < protocol["min_arm_confidence"]
                or not np.isfinite([*wrist, *elbow]).all()
                or np.linalg.norm(wrist - elbow) < 1
            ):
                continue
            palm = wrist + protocol["palm_forearm_fraction"] * (wrist - elbow)
            if not (0 <= palm[0] < frame["width"] and 0 <= palm[1] < frame["height"]):
                continue
            hand["observed_wrist_xy"] = hand["xy"]
            hand["xy"] = palm.tolist()
            hand["position_basis"] = "forearm_extrapolation_not_observed_palm"
            hand["uncertainty"] = hand.get("uncertainty", []) + [
                "Palm can bend relative to forearm; projected center is not a detection."
            ]
            accepted.append(hand)
        frame["hands"] = accepted
    return result


def seed_texture(gray, bbox, protocol=PROTOCOL):
    """Use only corners strictly inside an actual detector box, never around it."""
    x1, y1, x2, y2 = np.rint(bbox).astype(int)
    h, w = gray.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None, "cash_box_too_small"
    mask = np.zeros_like(gray)
    mask[y1 + 1 : y2 - 1, x1 + 1 : x2 - 1] = 255
    pts = cv2.goodFeaturesToTrack(gray, 30, 0.02, 2, mask=mask, blockSize=3)
    if pts is None or len(pts) < protocol["flow_min_points"]:
        return None, "too_few_cash_texture_points"
    extent = np.ptp(pts[:, 0], axis=0) / [x2 - x1, y2 - y1]
    if min(extent) < protocol["flow_min_seed_extent_fraction"]:
        return None, "texture_not_distributed_over_cash_box"
    return pts, None


def follow_texture(previous, current, points, protocol=PROTOCOL):
    """Forward/backward LK consistency with robust translation, no reacquisition."""
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, winSize=(15, 15), maxLevel=2
    )
    if next_pts is None:
        return None, None, "flow_failed"
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, next_pts, None, winSize=(15, 15), maxLevel=2
    )
    if back is None:
        return None, None, "backward_flow_failed"
    valid = (
        status[:, 0].astype(bool)
        & back_status[:, 0].astype(bool)
        & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) <= protocol["flow_fb_error_px"])
        & (error[:, 0] <= protocol["flow_max_median_error"])
        & np.isfinite(next_pts[:, 0]).all(axis=1)
    )
    if valid.sum() < protocol["flow_min_points"]:
        return None, None, "too_few_consistent_texture_points"
    delta = next_pts[valid, 0] - points[valid, 0]
    translation = np.median(delta, axis=0)
    keep = (
        np.linalg.norm(delta - translation, axis=1) <= protocol["flow_max_translation_residual_px"]
    )
    chosen = next_pts[valid][keep]
    if len(chosen) < protocol["flow_min_points"]:
        return None, None, "incoherent_texture_motion"
    return chosen, np.median(delta[keep], axis=0), None


def template_similarity(gray, bbox, template):
    x1, y1, x2, y2 = np.rint(bbox).astype(int)
    h, w = gray.shape
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
        return None
    patch = cv2.resize(gray[y1:y2, x1:x2], (template.shape[1], template.shape[0]))
    if min(patch.std(), template.std()) < 2:
        return None
    return float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
