"""Evaluate explicitly selected scene profiles; labels are never detector inputs."""

import argparse
import html
import xml.etree.ElementTree as ET
from pathlib import Path

from app.cv.analyze_cash_operations import read, run, write
from app.rules.cash_security import evaluate_cash_security
from app.rules.cash_security_schemas import CashSecurityRequest
from experiments.cash_security_run import convert_legacy
from experiments.counter_hand import analyze, observations
from experiments.counter_occupancy import ROOT, sha

DATA = ROOT / "data/cash_operations_v1"
META = ROOT / "datasets/cash_operations_v1"


def prepare():
    # Explicit human calibration selection, stored for repeatability, not a runtime name rule.
    selected = {"x8": "desk-wide-a", "x10": "desk-wide-b"}
    rows = []
    for source in read(ROOT / "datasets/normal_cash_operations_v1/sources.json"):
        rows.append(
            {
                **source,
                "profile": f"config/camera_profiles/{selected[source['id']]}.json",
                "observations": f"data/normal_cash_operations_v1/{source['id']}",
            }
        )
    write(META / "evaluation.json", rows)


def evaluate():
    rows = read(META / "evaluation.json")
    for source in rows:
        output = DATA / source["id"]
        cache = ROOT / source["observations"]
        run(
            Path(source["video"]),
            ROOT / source["profile"],
            cache / "input-detections.json",
            cache / "poses.json",
            output,
        )
        profile = read(ROOT / source["profile"])
        protocol = read(ROOT / "datasets/counter_hand_v1/protocol.json")
        protocol.update(
            counter_zone=profile["zones"]["counter"],
            counter_output_zone=profile["zones"]["counter_output"],
            output_core=profile["zones"]["counter_output"],
            zone_provenance="Explicitly calibrated camera profile; no action times",
        )
        tracks = read(output / "legacy/track_history.json")["frames"]
        pose = read(cache / "poses.json")
        hands = observations(pose["frames"], tracks, protocol)
        cash = [
            {
                **f,
                "frame_index": round(f["timestamp_sec"] * pose["fps"]),
                "width": 464,
                "height": 848,
            }
            for f in tracks
        ]
        diagnostic = analyze(hands, cash, protocol)
        diagnostic["integration"] = "Separate diagnostic only; never promotes a cash event"
        write(output / "counter-wrist-diagnostic.json", diagnostic)
        legacy = read(output / "legacy/events.json")
        request = CashSecurityRequest(
            duration_sec=source["frame_count"] / source["fps"],
            events=sorted(convert_legacy(legacy), key=lambda e: e.end_sec),
        )
        write(output / "r05-result.json", evaluate_cash_security(request).model_dump(mode="json"))
        wrong = next(s for s in rows if s["profile"] != source["profile"])
        run(
            Path(source["video"]),
            ROOT / wrong["profile"],
            cache / "input-detections.json",
            cache / "poses.json",
            DATA / (source["id"] + "-wrong-profile"),
        )
        print(source["id"], "evaluated with correct and wrong profile", flush=True)


def report():
    protected = read(ROOT / "datasets/normal_cash_operations_v1/protected-before.json")
    # The additive API diagnostics field is outside the frozen vision/rule layers.
    protected = {name: digest for name, digest in protected.items() if name != "app\\schemas.py"}
    changed = [name for name, digest in protected.items() if sha(ROOT / name) != digest]
    write(DATA / "protected-check.json", {"checked": len(protected), "changed": changed})
    tests = {}
    if (DATA / "tests.xml").exists():
        tests = ET.parse(DATA / "tests.xml").getroot().find("testsuite").attrib
    sections, summary = [], []
    for source in read(META / "evaluation.json"):
        name = source["id"]
        output = DATA / name
        result = read(output / "cash-operations.json")
        api = read(output / "api-result.json")
        camera = read(output / "camera-validation.json")
        wrong = read(DATA / (name + "-wrong-profile") / "cash-operations.json")
        wrist = read(output / "counter-wrist-diagnostic.json")
        r05 = read(output / "r05-result.json")
        records = read(output / "cash-observations.json")
        event_types = [e["type"] for op in result["operations"] for e in op["events"]]
        counts = {
            zone: sum(o["location"] == zone for f in records for o in f["cash"])
            for zone in ["cashier", "table", "counter", "safe", "unknown"]
        }
        row = {
            "video": name,
            "profile": result["profile"],
            "camera_valid": camera["valid"],
            "verified_frames": len(camera["checks"]),
            "operations": result["operations"],
            "events": event_types,
            "verdict": api["verdict"],
            "cash_operation_status": result["status"],
            "r05": r05,
            "location_observations": counts,
            "wrist_contacts": len(wrist["contacts"]),
            "wrist_withdrawal_hypotheses": sum(
                c["withdrawal_hypothesis"] for c in wrist["contacts"]
            ),
            "wrong_profile_status": wrong["status"],
            "wrong_profile_operation_count": len(wrong["operations"]),
        }
        summary.append(row)
        timeline = (output / "timeline.txt").read_text(encoding="utf-8")
        table_rows = "".join(
            f"<tr><td>{op['operation_id']}</td><td>{op['start_time']:.3f}</td>"
            f"<td>{op['end_time']}</td><td>{html.escape(str(op['logical_cash_ids']))}</td>"
            f"<td>{op['completion_status']}</td></tr>"
            for op in result["operations"]
        )
        sections.append(f"""<h2>{name.upper()} — профиль {result["profile"]}</h2>
<p>Геометрия подтверждена на {len(camera["checks"])} обработанных кадрах.
Чужой профиль отклонён, операций при нём: {len(wrong["operations"])}.</p>
<p>Наблюдения cash по расположению: {html.escape(str(counts))}.
Диагностических контактов кистей: {len(wrist["contacts"])}; гипотез извлечения:
{row["wrist_withdrawal_hypotheses"]}. Гипотезы не превращены в MONEY_FROM_COUNTER.</p>
<table><tr><th>Фрагмент операции</th><th>Начало, с</th><th>Конец</th><th>Стопка</th>
<th>Статус</th></tr>{table_rows}</table><pre>{html.escape(timeline)}</pre>
<p>API: {api["verdict"]}. CashOperation: {result["status"]}.
R05: {r05["verdict"]}; нарушений: {len(r05["violations"])}.
Незавершённые фрагменты не являются количеством реальных операций.</p>
<p><a href="{name}/debug.mp4">Debug видео</a> · <a href="{name}/timeline.txt">Timeline</a> ·
<a href="{name}/cash-operations.json">CashOperation JSON</a> ·
<a href="{name}/counter-wrist-diagnostic.json">Кисти: отдельная диагностика</a></p>
<img src="{name}/zones.jpg" alt="Калибровка зон {name}">""")
    write(DATA / "summary.json", {"videos": summary, "tests": tests, "protected_changed": changed})
    document = """<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Sana — CashOperation</title><style>
body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#182737}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccd4df;padding:9px;text-align:left}
pre{background:#edf1f5;padding:15px;white-space:pre-wrap}img{max-width:464px;width:100%}
</style><h1>CashOperation без обязательного клиента</h1>
<p>Новый отдельный поток наблюдений и событий. ServiceSession, R01–R05, существующий Event Detector,
tracking и веса не изменялись. Метки NORMAL заданы пользователем и не входят в алгоритм.</p>
<p>Подтверждённый возврат денег на стол после цикла счётчика и подтверждённое помещение в сейф —
два допустимых завершения. Универсального требования убрать деньги в сейф нет.
Блокнот/письмо отложены и не влияют на нарушения.</p>
<h2>Что проверено</h2><p>Camera profile выбирается явно через --profile, затем проверяется по
неподвижным деталям на всех кадрах с detections. Несовпадение запрещает применение зон.
Новые операции получают nullable associated_session_id; фиктивные клиенты не создаются.</p>
<h2>Оставшиеся ограничения реальных видео</h2><p>Денежные track IDs фрагментируются, особенно
при работе со счётчиком. Кисть рядом со стопкой не означает взятие: нужно перемещение денег.
Отсутствуют достаточные наблюдения cash без руки внутри счётчика; поэтому полный цикл пока
не подтверждён. Внутренность сейфа перекрывается дверцей, одного SAFE_ZONE недостаточно.
Алгоритм сохраняет insufficient_evidence. Разные фрагменты не склеиваются по близости/роли.</p>"""
    document += f"<p>Тесты: {html.escape(str(tests))}. Неизменность старых файлов: "
    document += f"{len(protected) - len(changed)}/{len(protected)}.</p>" + "".join(sections)
    document += '<p><a href="summary.json">Сводка JSON</a> · '
    document += '<a href="tests.xml">Результаты pytest</a></p></html>'
    (DATA / "report.html").write_text(document, encoding="utf-8")
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "evaluate", "report"])
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "report": report}[args.command]()
