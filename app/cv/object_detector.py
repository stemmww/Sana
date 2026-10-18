"""Local cash/id_card detections without ownership or action inference."""

import math
from pathlib import Path

from app.cv.object_dataset import CLASSES, sha256
from app.cv.train_cash import local_yolo


class ObjectDetector:
    def __init__(self, weights: Path, confidence=0.25, imgsz=768, device="cpu"):
        if not math.isfinite(confidence) or not 0 < confidence <= 1 or imgsz < 32:
            raise ValueError("Invalid confidence or image size")
        self.model_sha256 = sha256(weights)
        self.model = local_yolo()(str(weights))
        if sha256(weights) != self.model_sha256:
            raise ValueError("Weights changed during loading")
        self.supported = set(self.model.names.values())
        if self.supported not in ({"cash"}, set(CLASSES)):
            raise ValueError("Expected cash-only baseline or cash/id_card candidate weights")
        self.confidence, self.imgsz, self.device = confidence, imgsz, device

    def detect(self, frame) -> list[dict]:
        result = self.model.predict(
            frame, conf=self.confidence, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        return [
            {"label": self.model.names[int(cls)], "box_xyxy": box, "confidence": score}
            for box, score, cls in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            )
        ]
