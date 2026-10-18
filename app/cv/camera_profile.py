"""Explicit scene profiles with a visual geometry gate, independent of video names."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


class CameraProfile:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))
        reference_path = path.parent / self.data["reference_image"]
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != self.data["reference_sha256"]:
            raise ValueError("Camera reference checksum mismatch")
        self.reference = cv2.imdecode(np.fromfile(reference_path, dtype=np.uint8), 1)
        if self.reference is None:
            raise ValueError("Camera reference unavailable")
        self.gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        for box in self.data["static_anchors_px"]:
            x1, y1, x2, y2 = box
            if self.gray[y1:y2, x1:x2].std() < 8:
                raise ValueError("Camera anchor lacks texture")

    def validate(self, frame):
        if frame is None or frame.shape != self.reference.shape:
            return {"valid": False, "reason": "camera_dimensions_mismatch", "anchors": []}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows = []
        for x1, y1, x2, y2 in self.data["static_anchors_px"]:
            template = self.gray[y1:y2, x1:x2]
            left, top = max(0, x1 - 6), max(0, y1 - 6)
            search = gray[top : min(gray.shape[0], y2 + 6), left : min(gray.shape[1], x2 + 6)]
            _, score, _, pos = cv2.minMaxLoc(
                cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            )
            shift = float(np.hypot(pos[0] + left - x1, pos[1] + top - y1))
            rows.append(
                {
                    "ncc": float(score),
                    "shift_px": shift,
                    "matched": bool(score >= 0.85 and shift <= 4),
                }
            )
        valid = sum(r["matched"] for r in rows) >= 3
        return {
            "valid": valid,
            "reason": None if valid else "camera_geometry_unverified",
            "anchors": rows,
        }

    def overlay(self, frame):
        canvas = frame.copy()
        h, w = frame.shape[:2]
        colors = [
            (0, 220, 0),
            (255, 200, 0),
            (0, 200, 255),
            (0, 0, 255),
            (255, 0, 255),
            (255, 120, 0),
        ]
        for (name, poly), color in zip(self.data["zones"].items(), colors, strict=True):
            points = np.rint(np.array(poly) * [w, h]).astype(np.int32)
            cv2.polylines(canvas, [points], True, color, 1)
            x, y = points[0]
            cv2.putText(canvas, name, (int(x), max(12, int(y) - 4)), 0, 0.35, color, 1)
        return canvas
