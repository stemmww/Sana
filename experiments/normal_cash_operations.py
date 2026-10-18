"""Isolated audit of operator-only cash operations; no production event changes.

User scenario labels are evaluation metadata, never input to detection or rules.
Run prepare, inspect, infer, replay, summary (in that order).
"""

from __future__ import annotations

import argparse
import html
from collections import Counter
from pathlib import Path

import cv2

from experiments import cash_security_video as video_audit
from experiments.counter_occupancy import ROOT, read_json, sha, write_json

DATA = ROOT / "data/normal_cash_operations_v1"
META = ROOT / "datasets/normal_cash_operations_v1"


def protected():
    paths = [*ROOT.joinpath("app").rglob("*.py"), *ROOT.joinpath("config").rglob("*.json")]
    paths += [ROOT / "data/cash/models/cash.pt", ROOT / "yolo11n.pt"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def prepare():
    # Filenames select evaluation clips only. All algorithms use identical settings.
    sources = []
    videos = Path.home() / "Downloads/open/open/videos"
    for number in (8, 10):
        matches = [p for p in videos.glob("*.mov") if p.stem.split("-")[-1] == str(number)]
        if len(matches) != 1:
            raise ValueError(f"Expected one evaluation clip {number}, found {matches}")
        path = matches[0]
        cap = cv2.VideoCapture(str(path))
        sources.append(
            {
                "id": f"x{number}",
                "video": str(path),
                "video_sha256": sha(path),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
                "dimensions": [
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                ],
            }
        )
        cap.release()
        (DATA / f"x{number}").mkdir(parents=True, exist_ok=True)
    write_json(META / "sources.json", sources)
    write_json(
        META / "expected-scenarios.json",
        {
            "source": "user description; evaluation metadata only, not detector input",
            "scenarios": [
                {
                    "id": "x8",
                    "expected_label": "NORMAL",
                    "customer_required": False,
                    "expected_actions": [
                        "cash placed in counter",
                        "counted cash retrieved",
                        "cash placed in safe",
                        "notebook taken",
                        "writing in notebook",
                    ],
                },
                {
                    "id": "x10",
                    "expected_label": "NORMAL",
                    "customer_required": False,
                    "expected_actions": [
                        "cash stack on table",
                        "one stack placed in counter",
                        "counted cash retrieved",
                        "counted cash placed on table",
                    ],
                },
            ],
            "interpretation": (
                "Returning cash to the table during an operation does not alone satisfy R05."
            ),
        },
    )
    baseline = META / "protected-before.json"
    if not baseline.exists():
        write_json(baseline, protected())
    print("Prepared isolated evaluation sources", flush=True)


def replay():
    from app.cv.analyze_events import analyze_video

    for source in read_json(META / "sources.json"):
        output = DATA / source["id"]
        result = analyze_video(
            video=Path(source["video"]),
            output=output / "legacy",
            config_path=ROOT / "config/event_camera_service.json",
            replay=output / "input-detections.json",
        )
        print(source["id"], result, flush=True)


def inspect():
    from PIL import Image, ImageDraw

    video_audit.inspect()
    # Fixed display crop only, chosen from camera geometry; never used for inference.
    for source in read_json(META / "sources.json"):
        output = DATA / source["id"]
        frames = sorted((output / "review-frames").glob("*.jpg"))
        for start in range(0, len(frames), 9):
            batch = frames[start : start + 9]
            sheet = Image.new("RGB", (1485, 400 * ((len(batch) + 2) // 3)), "white")
            for index, path in enumerate(batch):
                x, y = (index % 3) * 495, (index // 3) * 400
                with Image.open(path) as original:
                    tile = original.crop((0, 300, 330, 550)).resize((495, 375))
                sheet.paste(tile, (x, y + 25))
                ImageDraw.Draw(sheet).text(
                    (x + 5, y + 5), f"{int(path.stem) / source['fps']:.2f} sec", fill="black"
                )
            sheet.save(output / f"detail-{start // 9}.jpg", quality=92)


def summary():
    from app.rules.cash_security import evaluate_cash_security
    from app.rules.cash_security_schemas import CashSecurityRequest
    from experiments.cash_security_run import convert_legacy

    rows = []
    for source in read_json(META / "sources.json"):
        output = DATA / source["id"]
        raw = read_json(output / "input-detections.json")
        doc = read_json(output / "legacy/events.json")
        tracks = read_json(output / "legacy/track_history.json")["frames"]
        events = doc["rule_request"]["events"]
        security_events = sorted(convert_legacy(doc), key=lambda e: e.end_sec)
        security = evaluate_cash_security(
            CashSecurityRequest(
                duration_sec=source["frame_count"] / source["fps"],
                events=security_events,
            )
        ).model_dump(mode="json")
        write_json(output / "r05-result.json", security)
        rows.append(
            {
                "video_id": source["id"],
                "duration_sec": source["frame_count"] / source["fps"],
                "sampled_frames": len(raw["frames"]),
                "frames_with_cash_detection": sum(bool(f["objects"]) for f in raw["frames"]),
                "cash_detections": sum(len(f["objects"]) for f in raw["frames"]),
                "person_role_observations": dict(
                    Counter(p["role"] for f in tracks for p in f["people"])
                ),
                "logical_cash_ids": sorted(
                    {o["logical_object_id"] for f in tracks for o in f["objects"]}
                ),
                "events": events,
                "rule_result": doc["rule_result"],
                "r05_result": security,
                "automatic_scenario_result": "not_implemented",
                "limitations": [
                    "Cash semantics require an active customer in the existing EventDetector.",
                    "Existing polygons do not align with these images; SAFE_ZONE is on the wall.",
                    "Safe observations require validated visual evidence.",
                    "Notebook and writing recognition are not implemented.",
                    "No violation detected is not automatic confirmation of NORMAL.",
                ],
            }
        )
    before, after = read_json(META / "protected-before.json"), protected()
    changes = [name for name in before.keys() | after.keys() if before.get(name) != after.get(name)]
    write_json(DATA / "protected-check.json", {"checked": len(before), "changed": changes})
    write_json(DATA / "summary.json", rows)
    for row in rows:
        print(
            row["video_id"],
            "cash",
            row["cash_detections"],
            "events",
            len(row["events"]),
            "sessions",
            len(row["rule_result"]["sessions"]),
            flush=True,
        )
    if changes:
        raise RuntimeError(f"Protected files changed: {changes}")
    manual = read_json(META / "manual-review.json")
    sections = []
    for row in rows:
        name = row["video_id"]
        timeline = (DATA / name / "legacy/timeline.txt").read_text(encoding="utf-8")
        notes = manual["videos"][name]
        actions = "".join(
            f"<tr><td>{html.escape(a['interval'])}</td><td>{html.escape(a['observation'])}</td>"
            f"<td>{html.escape(a['limitation'])}</td></tr>"
            for a in notes["actions"]
        )
        pictures = "".join(
            f'<a href="{name}/{p.name}"><img loading="lazy" src="{name}/{p.name}"></a>'
            for p in sorted((DATA / name).glob("detail-*.jpg"))
        )
        sections.append(f"""<section><h2>{name.upper()} — ожидаемая метка NORMAL</h2>
<p>{row["duration_sec"]:.2f} с; cash обнаружен в {row["frames_with_cash_detection"]} из
{row["sampled_frames"]} обработанных кадров; всего bbox cash: {row["cash_detections"]}.
Это количество наблюдений модели, а не precision/recall.</p>
<p>ServiceSession: {len(row["rule_result"]["sessions"])}.
R05: {html.escape(security_verdict(row))}.
Автоматическое подтверждение полного сценария не реализовано.</p>
<h3>Ручной просмотр, не события алгоритма</h3><table><tr><th>Примерный интервал</th>
<th>Наблюдение</th><th>Ограничение</th></tr>{actions}</table>
<h3>Реальная автоматическая timeline</h3><pre>{html.escape(timeline)}</pre>
<p><a href="{name}/legacy/events.json">events.json</a> ·
<a href="{name}/legacy/track_history.json">tracks</a> ·
<a href="{name}/legacy/debug.mp4">debug.mp4 (старые зоны)</a> ·
<a href="{name}/r05-result.json">R05 JSON</a></p>
<details><summary>Кадры для проверки действий</summary>{pictures}</details></section>""")
    report = """<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Sana — X-8 / X-10</title><style>
body{font:16px/1.6 system-ui;margin:32px auto;max-width:1150px;padding:0 20px;color:#172332}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccd5df;padding:10px;text-align:left}
pre{background:#edf1f5;padding:16px;white-space:pre-wrap}img{width:100%;margin:12px 0}
section{margin:32px 0;border-top:2px solid #ccd5df}a{color:#174d9d}
</style><h1>X-8 и X-10: самостоятельные операции кассира</h1>
<p>Сценарии зарегистрированы как NORMAL по описанию пользователя. Метки и ручные интервалы
не поступают в детекторы. Запуск существующих person/cash/pose моделей выполнен на RTX 3050.
Новые модели не обучались. R01–R05, production tracking, Event Detector и веса не изменялись.</p>
<h2>Что мешает автоматическому распознаванию</h2><ol>
<li>Старые polygon zones не соответствуют этому ракурсу: SAFE_ZONE располагается на стене.
Нужна отдельная геометрическая калибровка камеры; подбирать зоны по ожидаемым событиям нельзя.</li>
<li>EventDetector.object_event() возвращает None без active_customer. SecurityEvent также требует
customer_id для подтверждённых денежных событий. В этих роликах кассир работает самостоятельно.</li>
<li>Детектора блокнота/письма нет. Нельзя заменять запись в блокнот событием ID_CHECK.</li>
<li>Возврат денег на стол в X-10 сам по себе допустим. R05 требует следующего клиента,
подтверждённого извлечения и происхождения денег.
Отсутствие MONEY_TO_SAFE ничего не доказывает.</li></ol>
<h2>Минимальный следующий шаг</h2><p>Откалибровать отдельный профиль камеры и подготовить
диагностическую CashOperation, которая может существовать без ServiceSession. Для X-8 проверять
счётчик → сейф → действие записи; для X-10 — стол → счётчик → стол, сохраняя разные стопки отдельно.
Наблюдения блокнота и письма добавлять отдельно с evidence/uncertainty. Не подавать фиктивного
клиента в действующие правила. Здесь интеграция не выполнялась.</p>""" + "".join(sections)
    report += f"<p>Контроль неизменности: {len(before)}/{len(before)} файлов. "
    report += '<a href="summary.json">Сводка JSON</a> · '
    report += '<a href="protected-check.json">SHA-проверка</a></p></html>'
    (DATA / "report.html").write_text(report, encoding="utf-8")


def security_verdict(row):
    result = row["r05_result"]
    return result["verdict"] + "; " + ", ".join(u.get("code", "") for u in result["uncertainties"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "inspect", "infer", "replay", "summary"])
    args = parser.parse_args()
    video_audit.DATA, video_audit.META = DATA, META
    {
        "prepare": prepare,
        "inspect": inspect,
        "infer": video_audit.infer,
        "replay": replay,
        "summary": summary,
    }[args.command]()


if __name__ == "__main__":
    main()
