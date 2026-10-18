"""Replay real observations and compare immutable rule results before API composition."""

import argparse
import html
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from app.cv.analysis_result import assemble_video_result
from app.cv.analyze_cash_operations import read, run, write
from app.cv.analyze_events import analyze_video, sha256
from app.rules.cash_security_schemas import CashSecurityRequest
from experiments.counter_occupancy import ROOT
from experiments.service_cash_validation import audit

DATA = ROOT / "data/verdict_separation_v1"
META = ROOT / "datasets/verdict_separation_v1"


def prepare():
    rows = []
    for source in read(ROOT / "datasets/counter_sequence_v1/sources.json"):
        if source["id"] == "x5":
            continue  # Its newer service-boundary validation is selected below.
        rows.append(
            {
                **source,
                "mode": "legacy",
                "config": "config/event_camera.json",
                "detections": source["cache"] + "/detections.json",
                "baseline": source["cache"],
            }
        )
    for source in read(ROOT / "datasets/service_boundaries_v1/sources.json"):
        rows.append(
            {
                **source,
                "mode": "service",
                "config": "config/event_camera_service.json",
                "baseline": "data/service_boundaries_v1/" + source["id"],
            }
        )
    for source in read(ROOT / "datasets/cash_operations_v1/evaluation.json"):
        rows.append(
            {
                **source,
                "mode": "cash_operation",
                "baseline": "data/cash_operations_v1/" + source["id"],
            }
        )
    write(META / "evaluation.json", rows)
    # These hashes come from before this task, not after edits.
    old = read(ROOT / "datasets/normal_cash_operations_v1/protected-before.json")
    protected = {k: v for k, v in old.items() if k != "app\\schemas.py"}
    write(META / "protected-before.json", protected)


def without_paths(value):
    if isinstance(value, dict):
        return {k: without_paths(v) for k, v in value.items() if k != "evidence_frames"}
    if isinstance(value, list):
        return [without_paths(v) for v in value]
    return value


def evaluate():
    for source in read(META / "evaluation.json"):
        video, output = Path(source["video"]), DATA / source["id"]
        if sha256(video) != source["video_sha256"]:
            raise ValueError("Source video changed")
        baseline = ROOT / source["baseline"]
        cash = None
        request = None
        if source["mode"] == "cash_operation":
            cache = ROOT / source["observations"]
            run(
                video,
                ROOT / source["profile"],
                cache / "input-detections.json",
                cache / "poses.json",
                output,
            )
            cash = read(output / "cash-operations.json")
        elif source["mode"] == "service":
            audit(
                {
                    **source,
                    "detections": str(ROOT / source["detections"]),
                    "auxiliary": str(ROOT / source["auxiliary"]),
                },
                output,
                ROOT / source["config"],
                replay=True,
            )
            request = CashSecurityRequest.model_validate(read(output / "security-request.json"))
        else:
            analyze_video(
                video,
                output / "legacy",
                ROOT / source["config"],
                replay=ROOT / source["detections"],
            )
        legacy = read(output / "legacy/events.json")
        api = assemble_video_result(video, legacy, cash, request)
        write(output / "api-result.json", api.model_dump(mode="json"))
        write(output / "semantic-events.json", api.diagnostics["semantic_events"])
        comparison = {}
        old_legacy = read(
            baseline / ("events.json" if source["mode"] == "legacy" else "legacy/events.json")
        )
        comparison["r01_r04_unchanged"] = legacy["rule_result"] == old_legacy["rule_result"]
        comparison["service_semantic_events_unchanged"] = legacy["events"] == old_legacy["events"]
        if source["mode"] != "legacy":
            old_r05 = read(
                baseline
                / ("security-result.json" if source["mode"] == "service" else "r05-result.json")
            )
            comparison["r05_unchanged"] = api.diagnostics["rule_reports"]["R05"] == old_r05
        if cash is not None:
            old_cash = read(baseline / "cash-operations.json")
            comparison["cash_operations_unchanged"] = without_paths(
                cash["operations"]
            ) == without_paths(old_cash["operations"])
        comparison["no_synthetic_events"] = all(
            e.get("source") not in {"synthetic_example", "manual_annotation"}
            for e in api.diagnostics["semantic_events"]
        )
        write(output / "regression-check.json", comparison)
        print(
            source["id"],
            api.verdict,
            "violations",
            len(api.violations),
            "cash",
            api.diagnostics["cash_operation_status"],
            comparison,
            flush=True,
        )
        if not all(comparison.values()):
            raise AssertionError(f"Unexpected recognition or rule change: {source['id']}")


def event_lines(events):
    lines = []
    for event in events:
        timestamp = event.get("timestamp", event.get("timestamp_sec", event.get("end_sec")))
        status = event.get("status", "confirmed" if event.get("accepted", True) else "uncertain")
        stream = event.get("event_stream", "service")
        confidence = event.get("confidence")
        lines.append(f"{timestamp:.3f} [{stream}] {event['type']} {status} confidence={confidence}")
    return lines


def report():
    protected = read(META / "protected-before.json")
    changed = [name for name, digest in protected.items() if sha256(ROOT / name) != digest]
    write(DATA / "protected-check.json", {"checked": len(protected), "changed": changed})
    summary, sections, table_rows = [], [], []
    for source in read(META / "evaluation.json"):
        name, output = source["id"], DATA / source["id"]
        api = read(output / "api-result.json")
        diag = api["diagnostics"]
        legacy = read(output / "legacy/events.json")
        real_types = Counter(e["type"] for e in legacy["events"])
        cash = diag["cash_operation"]
        cash_events = [e for op in (cash or {}).get("operations", []) for e in op["events"]]
        semantic_types = dict(real_types + Counter(e["type"] for e in cash_events))
        row = {
            "video": name,
            "verdict": api["verdict"],
            "violations": api["violations"],
            "cash_operation_status": diag["cash_operation_status"],
            "analysis_status": diag["analysis_status"],
            "insufficient_evidence_reasons": diag["insufficient_evidence_reasons"],
            "semantic_events": diag["semantic_events"],
            "service_and_cash_event_counts": semantic_types,
            "rule_verdicts": {k: v["verdict"] for k, v in diag["rule_reports"].items()},
            "regression": read(output / "regression-check.json"),
        }
        summary.append(row)
        lines = event_lines(diag["semantic_events"])
        (output / "semantic-timeline.txt").write_text("\n".join(lines), encoding="utf-8")
        table_rows.append(
            f"<tr><td>{name.upper()}</td><td>{api['verdict']}</td>"
            f"<td>{len(api['violations'])}</td><td>{diag['cash_operation_status']}</td>"
            f"<td>{html.escape(str(row['rule_verdicts']))}</td></tr>"
        )
        reasons = json.dumps(diag["insufficient_evidence_reasons"], ensure_ascii=False, indent=2)
        cash_text = (
            json.dumps(cash, ensure_ascii=False, indent=2)
            if cash
            else (
                "CashOperation не запускалась: регрессия использует прежний сценарий обслуживания."
            )
        )
        sections.append(f"""<h2>{name.upper()}</h2>
<p>verdict: <b>{api["verdict"]}</b>; violations: {len(api["violations"])};
CashOperation: {diag["cash_operation_status"]}; анализ: {diag["analysis_status"]}.</p>
<details><summary>Причины неполноты</summary><pre>{html.escape(reasons)}</pre></details>
<details><summary>CashOperation: начало, конец и события</summary>
<pre>{html.escape(cash_text)}</pre></details>
<h3>Реальная semantic timeline (потоки service / cash_security / cash_operation)</h3>
<pre>{html.escape(chr(10).join(lines))}</pre>
<p>Поток cash_security представляет те же наблюдения для R05.
Это не дополнительные физические действия.</p>
<p><a href="{name}/api-result.json">Ответ API и diagnostics</a> ·
<a href="{name}/semantic-timeline.txt">Timeline</a> ·
<a href="{name}/regression-check.json">Сравнение с прежним результатом</a> ·
<a href="{name}/legacy/debug.mp4">Debug video</a></p>""")
    tests = {}
    if (DATA / "tests.xml").exists():
        tests = ET.parse(DATA / "tests.xml").getroot().find("testsuite").attrib
    write(DATA / "summary.json", {"videos": summary, "tests": tests, "protected_changed": changed})
    document = """<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Sana: recognition status / rule result / API verdict</title><style>
body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#172736}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:9px}
pre{background:#edf1f5;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere}
</style><h1>Три независимых результата анализа</h1>
<ol><li>CashOperation.status — полнота восстановления денежного цикла.
insufficient_evidence сохраняется, end_time остаётся null при неподтверждённом конце.</li>
<li>Rule Engine — прежние verdict/violations/uncertainties каждого набора правил.</li>
<li>Внешний API — violation при наличии подтверждённой тревоги, иначе no_violation.</li></ol>
<p><b>no_violation означает отсутствие обнаруженных подтверждённых нарушений.
Это не доказательство соблюдения всех кассовых процедур и не автоматическая метка NORMAL.</b>
В частности, ожидаемые нарушения X-1/X-2/X-4 не подставляются из названий файлов.</p>
<p>Реальные видео повторно обработаны по сохранённым YOLO/pose observations с проверкой SHA.
Веса, tracking, геометрия, правила R01–R05 и критерии завершения CashOperation не менялись.
Новые synthetic events при обработке видео не создавались.</p>
<table><tr><th>Видео</th><th>API verdict</th><th>Violations</th><th>CashOperation</th>
<th>Исходные rule verdicts</th></tr>"""
    document += "".join(table_rows) + "</table>"
    document += f"<p>Регрессионные тесты: {html.escape(str(tests))}. "
    document += f"Контроль неизменности: {len(protected) - len(changed)}/{len(protected)}.</p>"
    document += "".join(sections) + "</html>"
    (DATA / "report.html").write_text(document, encoding="utf-8")
    if changed:
        raise AssertionError(f"Protected code/weights changed: {changed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "evaluate", "report"])
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "report": report}[args.command]()
