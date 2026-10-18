"""Build audit, synthetic API examples and an HTML report from actual run artifacts."""
# HTML rows are literal report text.
# ruff: noqa: E501

from __future__ import annotations

import html
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from experiments.cash_security_video import DATA, META
from experiments.counter_occupancy import ROOT, imread, read_json, sha, write_json


def main():
    baseline = read_json(META / "baseline-sha256.json")
    changes = [
        name.replace("\\", "/")
        for name, digest in baseline.items()
        if sha(ROOT / name) != digest
    ]
    assert changes == ["app/api/routes.py"], changes
    tests = ET.parse(DATA / "tests.xml").getroot()
    cases = tests.findall(".//testcase")
    assert not tests.findall(".//failure") and not tests.findall(".//error")
    results = read_json(META / "results.json")
    model = read_json(META / "model-audit.json")
    assert not model["id_card_supported_by_loaded_cash_model"]
    counts = Counter(case.attrib["classname"] for case in cases)
    refs = []
    timeline_html, images = [], []
    for source in read_json(META / "sources.json"):
        v = source["id"]
        out = DATA / v
        events = read_json(out / "security-events.json")
        refs += [p for e in events for p in e["evidence_frames"]]
        longest = max(
            (e for e in events if e["type"] == "MONEY_LEFT_ON_TABLE"),
            key=lambda e: e["end_sec"] - e["start_sec"],
            default=None,
        )
        if longest:
            tracks = read_json(out / "legacy/track_history.json")["frames"]
            tiles = []
            for path in longest["evidence_frames"]:
                raw = imread(ROOT / path)
                index = int(Path(path).stem)
                fps = read_json(out / "poses.json")["fps"]
                frame = min(tracks, key=lambda f: abs(f["timestamp_sec"] - index / fps))
                for obj in frame["objects"]:
                    box = [
                        round(n * (464 if i % 2 == 0 else 848)) for i, n in enumerate(obj["bbox"])
                    ]
                    cv2.rectangle(raw, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 1)
                    cv2.putText(
                        raw,
                        f"{obj['logical_object_id']} {obj['confidence']:.2f}",
                        tuple(box[:2]),
                        0,
                        0.35,
                        (0, 0, 255),
                        1,
                    )
                crop = raw[340:600, :430]
                tile = Image.new("RGB", (430, 285), "white")
                ImageDraw.Draw(tile).text(
                    (5, 5), f"{v} {index / fps:.2f}s | table candidate: insufficient", fill="black"
                )
                tile.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (0, 25))
                tiles.append(tile)
            sheet = Image.new("RGB", (430 * len(tiles), 285), "white")
            for n, tile in enumerate(tiles):
                sheet.paste(tile, (n * 430, 0))
            sheet.save(out / "table-evidence.jpg", quality=93)
            images.append(
                f'<h3>{v.upper()}: пример наблюдений на столе</h3><a href="{v}/table-evidence.jpg"><img src="{v}/table-evidence.jpg"></a>'
            )
        lines = [
            f"{e['start_sec']:.2f}–{e['end_sec']:.2f} с | {e['type']} | {e['status']} | {e['confidence']:.3f}"
            for e in events
        ]
        timeline_html.append(
            f'<h3>{v.upper()}</h3><pre>{html.escape(chr(10).join(lines))}</pre><p><a href="{v}/timeline.txt">Полный timeline с причинами</a> · <a href="{v}/security-events.json">Events + evidence</a> · <a href="{v}/security-result.json">R05 и State Machine</a> · <a href="{v}/legacy/debug.mp4">Существующий детектор: debug.mp4</a></p>'
        )
    assert all((ROOT / p).exists() for p in refs)
    write_json(
        META / "audit.json",
        {
            "baseline_files": len(baseline),
            "existing_files_changed": changes,
            "unchanged_files": len(baseline) - len(changes),
            "verified_evidence_references": len(refs),
            "tests_passed": len(cases),
            "tests_by_module": dict(counts),
            "production_weights_unchanged": True,
            "legacy_rule_engine_unchanged": True,
            "legacy_event_detector_unchanged": True,
            "legacy_tracking_unchanged": True,
            "new_source_sha256": {
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    *ROOT.glob("app/rules/cash_security*.py"),
                    ROOT / "app/events/cash_security.py",
                    ROOT / "config/cash_security_camera.json",
                ]
            },
        },
    )
    # These are explicitly synthetic request examples, separate from real video results.
    from tests.test_cash_security import base, boundary, event, table

    normal = (
        base()
        + [
            event(k, t)
            for t, k in enumerate(
                [
                    "SAFE_OPENED",
                    "MONEY_TO_SAFE",
                    "SAFE_CLOSED",
                    "SAFE_LOCKED_VISUAL",
                    "ID_RETURNED_TO_CUSTOMER",
                    "CUSTOMER_EXIT",
                ],
                start=9,
            )
        ]
        + [boundary("SERVICE_STARTED")]
    )
    violation = base() + [event("CUSTOMER_EXIT", 14), boundary(), table()]
    for name, events in [("normal", normal), ("violation", violation)]:
        write_json(
            ROOT / f"config/cash_security_examples/{name}.json",
            {"duration_sec": 20, "events": sorted(events, key=lambda e: e["end_sec"])},
        )
    rows = "".join(
        f"<tr><td>{r['video'].upper()}</td><td>{'NORMAL' if r['video'] == 'x5' else 'VIOLATION'}</td><td>{r['verdict']}</td><td>{r['violations']}</td><td>{r['wrist_withdrawal_hypotheses']}, неподтверждённая</td><td>{r['id_detections']}</td></tr>"
        for r in results
    )
    doc = (
        """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sana — R05, сейф и ID</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1150px;margin:36px auto;padding:0 20px;color:#202b36}h1,h2,h3{line-height:1.3}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #ccd5df;padding:10px;text-align:left}th{background:#edf3f8}pre{padding:14px;background:#edf3f8;white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;height:auto}.note{padding:16px;border-left:5px solid #d49b27;background:#fff4d7}a{color:#125c9b}</style>
<h1>R05: деньги предыдущей операции на столе</h1>
<p>Добавлены отдельный Rule Engine для R05, State Machine и модуль шести новых semantic events. Существующие R01–R04, сценарии X‑1/X‑2/X‑3, Event Detector, tracking, конфигурация event_camera.json и веса сохранены.</p>
<h2>Результаты реальных видео</h2><table><tr><th>Видео</th><th>Ожидание пользователя</th><th>Автоматический результат</th><th>R05 alerts</th><th>Гипотеза извлечения</th><th>ID detections</th></tr>"""
        + rows
        + """</table>
<p class="note">Ожидаемый R05 в X‑4 пока не подтверждён автоматически. X‑5 также не объявлен NORMAL. Отказ от решения в обоих случаях обусловлен пропусками наблюдений; отсутствие тревоги не считается правильной классификацией. Имена роликов и ожидаемые ответы не входят в правило или детектор.</p>
<h2>Где цепочки ломаются</h2><table><tr><th>Слой</th><th>X‑5</th><th>X‑4</th></tr>
<tr><td>Извлечение из счётчика</td><td>Запястье: 31,51–32,31 с, затем cash. Экспериментальная гипотеза; общего money_group_id нет.</td><td>Запястье: 31,46–32,26 с, затем cash. Такая же неопределённость происхождения денег.</td></tr>
<tr><td>Деньги на столе</td><td>Есть кандидаты при работе руками. Их неподвижность в 2D не доказывает размещение на поверхности.</td><td>Длительная детекция cash-22: 55,64–68,73 с (57 наблюдений) и 69,66–72,47 с. Связь с деньгами после счётчика не установлена.</td></tr>
<tr><td>Сейф</td><td>Изменение изображения двери: кандидат открытия 47,20–47,90 с, закрытия 50,94–51,64 с. Оба insufficient_evidence. Видимая траектория денег внутрь и действие ключом автоматически не подтверждены.</td><td>Подтверждённых действий сейфа нет. Это не доказательство нарушения.</td></tr>
<tr><td>Следующий клиент</td><td>person-1 сохраняется после смены клиента. Второго CUSTOMER_ENTER/SERVICE_STARTED нет.</td><td>person-3 сохраняется после смены клиента. Второго CUSTOMER_ENTER/SERVICE_STARTED нет.</td></tr>
<tr><td>R05</td><td>insufficient_evidence</td><td>insufficient_evidence: нет подтверждённой цепочки происхождения стопки и следующей операции.</td></tr></table>
<h2>ID: модель → detection → track → event</h2><p>Видеоадаптер загружает <code>data/cash/models/cash.pt</code> через <code>CashDetector</code>. Реально прочитанные классы: <code>{0: cash}</code>. Person-модель: <code>yolo11n.pt</code>. Для диагностики запястий отдельно загружена <code>yolo11s-pose.pt</code>; она не распознаёт ID. Новых моделей не обучали.</p>
<table><tr><th>Этап</th><th>X‑5</th><th>X‑4</th><th>Причина</th></tr><tr><td>id_card detection</td><td>0</td><td>0</td><td>Класса нет в загруженной модели. CashDetector выбирает только cash.</td></tr><tr><td>ID tracks</td><td>0</td><td>0</td><td>Нет входных ID detections; id_detector_validated=false.</td></tr><tr><td>ID_TRANSFER_TO_CASHIER</td><td>0</td><td>0</td><td>Нет подтверждённого перехода ID клиент → кассир.</td></tr><tr><td>ID_CHECK</td><td>0</td><td>0</td><td>Нет ID-track. Старый обработчик использует удержание как proxy проверки, не распознаёт чтение документа.</td></tr><tr><td>ID_RETURNED_TO_CUSTOMER</td><td>0</td><td>0</td><td>Новый модуль готов, но нет исходной передачи и того же ID при возврате.</td></tr></table>
<p>Простая замена cash.pt на двухклассовые веса недостаточна: старый CashDetector отфильтрует ID, а видеоадаптер запрещает id_detector_validated=true без валидированной интеграции. Существующие кандидаты cash/id_card не заменяли основные веса.</p>
<h2>State Machine</h2><pre>CASHIER_PRESENT → SERVICE_STARTED
→ ID_TRANSFER_TO_CASHIER → ID_CHECK
→ MONEY_TRANSFER_TO_CASHIER → MONEY_TO_COUNTER → MONEY_FROM_COUNTER
→ SAFE_OPENED → MONEY_TO_SAFE → SAFE_CLOSED → SAFE_LOCKED_VISUAL
→ ID_RETURNED_TO_CUSTOMER → CUSTOMER_EXIT → COMPLETED
Новая operation_id → новая последовательность без наследования ID/денег.</pre>
<p>Происхождение денег хранится отдельно от текущей сессии. R05 проверяет подтверждённое извлечение, положительные наблюдения той же группы денег на столе и следующий вход/новую сессию. Положительные кадры должны охватывать момент границы, с разрывом ≤0,8 с. Временное размещение и устаревшая последняя позиция не дают нарушения. Положительное MONEY_TO_SAFE исключает тревогу; противоречивые сведения об одной стопке требуют проверки. Таймкод R05 равен подтверждённому моменту новой границы; detected_at_sec отдельно показывает, когда собраны все свидетельства.</p>
<h2>Новые события и зоны</h2><p>Подготовлены SAFE_OPENED, MONEY_TO_SAFE, SAFE_CLOSED, SAFE_LOCKED_VISUAL, MONEY_LEFT_ON_TABLE, ID_RETURNED_TO_CUSTOMER. Модуль требует положительных фактов, нескольких кадров, надёжной принадлежности операции и подтверждённого источника. SAFE_LOCKED_VISUAL означает только видимое действие с ключом/замком. Недостаток этих данных сохраняется в status и uncertainty.</p>
<p>TABLE_ZONE скопирована без изменения. SAFE_ZONE и отдельный контур двери добавлены в <code>config/cash_security_camera.json</code> по статической геометрии камеры. Движение/различие картинки сейфа используется только для неподтверждённых кандидатов.</p>
<h2>Тесты и сохранность</h2><p>"""
        + f"{len(cases)} теста пройдены; {counts['tests.test_cash_security']} — новые проверки R05, semantic events и API, остальные — регрессия существующей системы. Есть одно предупреждение о deprecated httpx в Starlette, ошибок нет. SHA‑256: {len(baseline) - 1} прежних файлов сохранены; из существующих изменён только app/api/routes.py для добавления отдельного маршрута."
        + """</p>
<p>Проверены временное размещение, помещение в сейф, отсутствие наблюдения сейфа, неопределённое извлечение, другая стопка/клиент, разрыв наблюдений, противоречие сейфа и стола, отсутствие следующего клиента, смена operation_id, смена ID-карты, произвольные абсолютные таймкоды и новая нормальная последовательность.</p>
<h2>API и воспроизведение</h2><pre>POST /api/v1/rules/cash-security/evaluate
X-API-Key: действующий ключ Sana
Body: CashSecurityRequest {duration_sec, events, ...}</pre><p>Старый POST /api/v1/rules/evaluate сохраняет R01–R04. Новый маршрут явно выбирается вызывающей стороной; загрузка видео через jobs не получила автоматической подмены сценария. Synthetic-примеры: <a href="../../config/cash_security_examples/normal.json">normal.json</a> и <a href="../../config/cash_security_examples/violation.json">violation.json</a>.</p>
<pre>.\\.venv\\Scripts\\python.exe -m experiments.cash_security_video infer
.\\.venv\\Scripts\\python.exe -m experiments.cash_security_video replay
.\\.venv\\Scripts\\python.exe -m experiments.cash_security_run
.\\.venv\\Scripts\\python.exe -m experiments.cash_security_report</pre>
<h2>Реальные timelines</h2>"""
        + "".join(timeline_html)
        + "".join(images)
        + """<h3>X‑5: визуальная проверка сейфа</h3><img src="x5/safe-review.jpg" alt="Дверь сейфа открывается и закрывается; отдельная визуальная проверка"><p>Кадры этой ручной проверки не использовались для принятия автоматических решений или обучения.</p>
<h2>Следующий минимальный шаг</h2><p>Проверить границы обслуживания на смене клиента и сохранение происхождения одной стопки через разрывы cash-track в отдельном режиме. Затем сравнить существующий cash/id_card-кандидат и подготовить явный ID-адаптер с проверкой качества. Для сейфа сначала подтвердить видимость помещения денег и ключевого действия; отсутствие detection не превращать в подтверждение.</p><p><a href="../../datasets/cash_security_v1/model-audit.json">Аудит моделей</a> · <a href="../../datasets/cash_security_v1/results.json">Результаты JSON</a> · <a href="../../datasets/cash_security_v1/audit.json">SHA‑256 и тесты</a> · <a href="../../datasets/cash_security_v1/review.json">Отдельная визуальная проверка</a> · <a href="tests.xml">JUnit XML</a></p></html>"""
    )
    (DATA / "report.html").write_text(doc, encoding="utf-8")
    print(
        "Report", len(cases), "tests;", len(refs), "evidence references; baseline changes:", changes
    )


if __name__ == "__main__":
    main()
