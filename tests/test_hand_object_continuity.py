"""Safety tests for the isolated diagnostic; synthetic inputs stay in tests."""

from copy import deepcopy

import cv2
import numpy as np

from experiments.hand_object_continuity import (
    follow_texture,
    palm_frames,
    seed_texture,
    template_similarity,
)
from experiments.run_hand_object_continuity import flow_ownership


def test_hand_proxy_preserves_actual_wrist_owner_identity_and_input():
    frames = [
        {
            "width": 100,
            "height": 100,
            "hands": [
                {
                    "xy": [30, 30],
                    "elbow": [20, 30, 0.9],
                    "arm_confidence": 0.9,
                    "owner": "unknown",
                    "track_id": "a",
                    "body_track_id": None,
                }
            ],
        }
    ]
    original = deepcopy(frames)
    changed = palm_frames(frames)
    assert frames == original
    hand = changed[0]["hands"][0]
    assert hand["xy"] == [35, 30]
    assert hand["observed_wrist_xy"] == [30, 30]
    assert hand["owner"] == "unknown" and hand["body_track_id"] is None
    assert hand["track_id"] == "a" and hand["uncertainty"]
    assert "not_observed" in hand["position_basis"]


def test_low_confidence_and_out_of_frame_projection_are_not_measurements():
    frames = [
        {
            "width": 100,
            "height": 100,
            "hands": [
                {"xy": [30, 30], "elbow": [20, 30, 0.1], "arm_confidence": 0.1},
                {"xy": [99, 30], "elbow": [20, 30, 0.9], "arm_confidence": 0.9},
                {"xy": [30, 30], "elbow": [30, 30, 0.9], "arm_confidence": 0.9},
            ],
        }
    ]
    assert palm_frames(frames)[0]["hands"] == []


def test_uniform_cash_box_cannot_borrow_texture_from_surrounding_table():
    gray = np.random.default_rng(4).integers(0, 255, (80, 80), dtype=np.uint8)
    gray[20:60, 20:60] = 90
    points, reason = seed_texture(gray, [25, 25, 55, 55])
    assert points is None and reason == "too_few_cash_texture_points"


def test_texture_tracks_real_pixel_translation():
    image = np.random.default_rng(7).integers(0, 255, (100, 100), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (100, 100))
    points, reason = seed_texture(image, [30, 30, 70, 70])
    assert reason is None
    moved, translation, reason = follow_texture(image, shifted, points)
    assert reason is None and len(moved) >= 4
    np.testing.assert_allclose(translation, [2, 1], atol=0.2)
    assert template_similarity(shifted, [32, 31, 72, 71], image[30:70, 30:70]) > 0.99


def test_occlusion_and_textureless_patches_do_not_confirm_continuity():
    image = np.random.default_rng(7).integers(0, 255, (100, 100), dtype=np.uint8)
    occluded = np.full_like(image, 100)
    points, _ = seed_texture(image, [30, 30, 70, 70])
    _, _, reason = follow_texture(image, occluded, points)
    assert reason is not None
    assert template_similarity(occluded, [30, 30, 70, 70], image[30:70, 30:70]) is None


def flow_fixture(hand_movement=4, owner="customer"):
    samples, hands = [], []
    for index, t in enumerate([0.0, 0.15]):
        samples.append(
            {"timestamp_sec": t, "frame_index": index, "center_px": [30 + index * 4, 30]}
        )
        hands.append(
            {
                "timestamp_sec": t,
                "frame_index": index,
                "counter_width_px": 100,
                "hands": [
                    {
                        "xy": [30 + index * hand_movement, 30],
                        "owner": owner,
                        "body_track_id": "b",
                        "track_id": "hand",
                    }
                ],
            }
        )
    tracks = [
        {
            "cash_id": "same-id",
            "seed_owner": {"owner": "cashier"},
            "samples": samples,
            "uncertainty": [],
        }
    ]
    p = {"cash_wrist_distance_widths": 0.6, "cash_owner_margin_widths": 0.1, "hand_gap_sec": 0.35}
    return tracks, hands, p


def test_near_recipient_without_comotion_does_not_make_handover():
    tracks, hands, p = flow_fixture(hand_movement=0)
    flow_ownership(tracks, hands, p)
    assert tracks[0]["status"] == "insufficient_evidence"


def test_comotion_is_only_hypothesis_and_keeps_cash_id():
    tracks, hands, p = flow_fixture()
    flow_ownership(tracks, hands, p)
    assert tracks[0]["status"] == "handover_texture_hypothesis"
    assert tracks[0]["cash_id"] == "same-id"
    assert "events" not in tracks[0]


def test_unknown_owner_and_competing_people_cannot_create_transfer():
    tracks, hands, p = flow_fixture(owner="unknown")
    flow_ownership(tracks, hands, p)
    assert tracks[0]["status"] == "insufficient_evidence"
    tracks, hands, p = flow_fixture()
    for frame in hands:
        other = {**frame["hands"][0], "owner": "cashier", "body_track_id": "other"}
        frame["hands"].append(other)
    flow_ownership(tracks, hands, p)
    assert tracks[0]["status"] == "insufficient_evidence"


def test_missing_pose_cannot_supply_owner_to_texture_track():
    tracks, hands, p = flow_fixture()
    for frame in hands:
        frame["timestamp_sec"] += 10
    flow_ownership(tracks, hands, p)
    assert not tracks[0]["ownership_cues"]
    assert tracks[0]["status"] == "insufficient_evidence"
    tracks, _, p = flow_fixture()
    flow_ownership(tracks, [], p)
    assert tracks[0]["status"] == "insufficient_evidence"


def test_recipient_motion_without_cashier_seed_does_not_establish_transfer():
    tracks, hands, p = flow_fixture()
    tracks[0]["seed_owner"]["owner"] = "unknown"
    flow_ownership(tracks, hands, p)
    assert tracks[0]["status"] == "insufficient_evidence"
