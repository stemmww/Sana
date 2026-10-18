"""Connect the existing real-video pipeline to the worker's one-Path contract.

This adapter uses the configured fixed-camera zones. It does not promote any
experimental observations, choose rules by video identity, or change weights.
"""

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import Settings
from app.cv.analysis_result import assemble_video_result
from app.cv.analyze_events import analyze_video as analyze_events
from app.schemas import AnalysisResult

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


class VideoSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", env_prefix="SANA_", extra="ignore"
    )
    event_config: Path = Path("config/event_camera_service.json")
    person_weights: Path = Path("yolo11n.pt")
    cash_weights: Path = Path("data/cash/models/cash.pt")
    device: str = "auto"


def local_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def analyze_video(video: Path) -> AnalysisResult:
    started = time.perf_counter()
    settings, video_settings = Settings(), VideoSettings()
    config = local_path(video_settings.event_config)
    person = local_path(video_settings.person_weights)
    cash = local_path(video_settings.cash_weights)
    for path in [video, config, person, cash]:
        if not path.is_file():
            raise FileNotFoundError(f"Local analysis input is unavailable: {path}")
    device = select_device(video_settings.device)
    # Names are only artifact keys. No filename/timestamp/previous-result decisions.
    output = settings.storage_dir.resolve() / "analysis" / f"{video.stem}-{uuid4().hex[:8]}"
    logger.info("Starting real video inference on device %s", device)
    analyze_events(video, output, config, person, cash, device=device)
    legacy = json.loads((output / "events.json").read_text(encoding="utf-8"))
    # The separate cached-pose CashOperation experiment is not silently enabled.
    result = assemble_video_result(video, legacy)
    result.diagnostics["worker_analysis"] = {
        "processing_mode": "video_inference",
        "inference_device": device,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "sampled_frames": legacy["sampled_frames"],
        "camera_configuration": config.name,
        "camera_selection": "configured_fixed_camera; not_inferred_from_video_name",
        "camera_geometry_check": "not_run; zones_require_matching_camera_view",
        "id_detector": "unavailable_in_production_weights",
        "experimental_actions_integrated": False,
    }
    (output / "api-result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
