"""Freeze hand-calibrated scene geometry, not action annotations or event times."""

import shutil

from experiments.counter_occupancy import ROOT, read_json, sha, write_json


def main():
    target = ROOT / "config/camera_profiles"
    target.mkdir(parents=True, exist_ok=True)
    # Source names below select reference material only; runtime requires --profile.
    views = [
        (
            "desk-wide-a",
            "x8",
            {
                "cashier": [(100, 300), (231, 300), (231, 410), (100, 419)],
                "customer": [(0, 314), (46, 312), (46, 500), (0, 520)],
                "counter": [(48, 388), (88, 370), (105, 380), (117, 438), (53, 450)],
                "counter_output": [(99, 405), (112, 403), (116, 431), (102, 435)],
                "table": [(47, 438), (279, 380), (320, 401), (320, 428), (65, 497)],
                "safe": [(232, 301), (272, 309), (278, 394), (235, 405)],
            },
            [[16, 271, 70, 302], [299, 230, 324, 263], [278, 328, 296, 359], [152, 496, 195, 520]],
        ),
        (
            "desk-wide-b",
            "x10",
            {
                "cashier": [(75, 265), (200, 265), (200, 378), (75, 385)],
                "customer": [(0, 280), (27, 280), (27, 460), (0, 481)],
                "counter": [(7, 354), (47, 336), (64, 345), (83, 404), (15, 416)],
                "counter_output": [(64, 371), (77, 369), (81, 397), (67, 401)],
                "table": [(11, 402), (243, 351), (287, 369), (288, 396), (29, 460)],
                "safe": [(201, 267), (238, 277), (244, 361), (207, 371)],
            },
            [[4, 239, 60, 269], [272, 199, 293, 225], [242, 296, 263, 327], [124, 460, 169, 482]],
        ),
    ]
    for name, source, pixels, anchors in views:
        ref = ROOT / f"data/normal_cash_operations_v1/{source}/review-frames/00000.jpg"
        shutil.copyfile(ref, target / f"{name}.jpg")
        zones = {k: [[x / 464, y / 848] for x, y in v] for k, v in pixels.items()}
        cfg = read_json(ROOT / "config/event_camera.json")
        cfg["zones"].update({k: v for k, v in zones.items() if k in cfg["zones"]})
        cfg["zones"]["room"] = [[0, 0.29], [0.69, 0.29], [0.71, 0.68], [0, 0.68]]
        cfg["zones"]["exit"] = [[0, 0.26], [0.15, 0.26], [0.15, 0.40], [0, 0.43]]
        cfg["cashier_anchor"] = [
            sum(x for x, _ in pixels["cashier"]) / 1856,
            sum(y for _, y in pixels["cashier"]) / 3392,
        ]
        cfg["reflection_regions"] = [[[0, 0], [0.46, 0], [0.46, 0.30], [0, 0.32]]]
        write_json(
            target / f"{name}.json",
            {
                "profile_id": name,
                "reference_image": f"{name}.jpg",
                "reference_sha256": sha(target / f"{name}.jpg"),
                "selection": (
                    "Explicit --profile after matching reference preview; 3/4 static anchors "
                    "must validate on every frame used for a cash observation. "
                    "A mismatched view is rejected; never infer profile from video filename."
                ),
                "dimensions": [464, 848],
                "static_anchors_px": anchors,
                "zones": zones,
                "event_config": cfg,
                "safe_interior_validated": False,
                "customer_zone_scope": "Visible strip at customer side; not a complete entry gate",
            },
        )


if __name__ == "__main__":
    main()
