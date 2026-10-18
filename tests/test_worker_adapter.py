"""Worker bridge must run inference and retain genuine result diagnostics."""

import json
import sys
from types import SimpleNamespace

import pytest

from app.cv import worker_adapter as adapter
from app.schemas import AnalysisResult


@pytest.mark.parametrize("available,expected", [(True, "0"), (False, "cpu")])
def test_device_auto_uses_available_cuda(monkeypatch, available, expected):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
    )
    assert adapter.select_device("auto") == expected
    assert adapter.select_device("cpu") == "cpu"


def test_worker_bridge_does_not_return_cli_summary_or_promote_hypotheses(tmp_path, monkeypatch):
    paths = [tmp_path / n for n in ["arbitrary.mov", "zones.json", "person.pt", "cash.pt"]]
    for p in paths:
        p.write_bytes(b"test")
    video, config, person, cash = paths
    monkeypatch.setattr(adapter, "Settings", lambda: SimpleNamespace(storage_dir=tmp_path))
    monkeypatch.setattr(
        adapter,
        "VideoSettings",
        lambda: SimpleNamespace(
            event_config=config, person_weights=person, cash_weights=cash, device="cpu"
        ),
    )
    called = []

    def inference(v, out, cfg, pw, cw, **kwargs):
        called.append((v, cfg, pw, cw, kwargs))
        out.mkdir(parents=True)
        (out / "events.json").write_text(json.dumps({"sampled_frames": 5, "events": []}))
        return {"verdict": "insufficient_evidence", "output": str(out)}

    monkeypatch.setattr(adapter, "analyze_events", inference)

    def compose(v, legacy):
        assert v == video and legacy["sampled_frames"] == 5
        return AnalysisResult(
            verdict="no_violation",
            violations=[],
            diagnostics={
                "analysis_status": "incomplete",
                "cash_operation_status": "not_evaluated",
                "insufficient_evidence_reasons": [{"code": "cash_operation_not_evaluated"}],
                "semantic_events": [],
            },
        )

    monkeypatch.setattr(adapter, "assemble_video_result", compose)
    result = adapter.analyze_video(video)
    assert called == [(video, config, person, cash, {"device": "cpu"})]
    assert result.verdict == "no_violation" and not result.violations
    assert result.diagnostics["analysis_status"] == "incomplete"
    assert result.diagnostics["insufficient_evidence_reasons"]
    assert not result.diagnostics["semantic_events"]
    assert result.diagnostics["worker_analysis"]["processing_mode"] == "video_inference"
    saved = list((tmp_path / "analysis").glob("*/api-result.json"))
    assert len(saved) == 1
    assert AnalysisResult.model_validate_json(saved[0].read_text()) == result


def test_missing_weight_fails_before_inference(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pt"
    video = tmp_path / "input.mov"
    video.write_bytes(b"video")
    monkeypatch.setattr(adapter, "Settings", lambda: SimpleNamespace(storage_dir=tmp_path))
    monkeypatch.setattr(
        adapter,
        "VideoSettings",
        lambda: SimpleNamespace(
            event_config=video, person_weights=missing, cash_weights=missing, device="cpu"
        ),
    )
    with pytest.raises(FileNotFoundError):
        adapter.analyze_video(video)
    assert not (tmp_path / "analysis").exists()
