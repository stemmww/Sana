"""Export reviewable evidence and a report without changing diagnostic decisions."""

from __future__ import annotations

import json
from html import escape

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experiments.counter_occupancy import ROOT, imread, imwrite, read_json, sha, write_json
from experiments.counter_sequence_data import DATA, META


def html_table(headers, rows):
    return (
        "<table><tr>"
        + "".join(f"<th>{escape(str(x))}</th>" for x in headers)
        + "</tr>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>" for row in rows
        )
        + "</table>"
    )


def export():
    cfg = read_json(ROOT / "config/event_camera.json")
    summaries = []
    for source in read_json(META / "sources.json"):
        out = DATA / source["id"]
        data = read_json(out / "features.json")
        variants = {
            mode: read_json(out / f"{mode}-events.json")
            for mode in ["body_box_reach", "zone_motion_only"]
        }
        wanted = {
            r["frame_index"]
            for pred in variants.values()
            for e in pred["events"]
            for r in e["evidence_frames"]
        }
        states = {f["frame_index"]: f for f in data["frames"]}
        cap = cv2.VideoCapture(source["video"])
        debug = cv2.VideoCapture(str(ROOT / source["cache"] / "debug.mp4"))
        writer = None
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index not in states:
                index += 1
                continue
            f = states[index]
            h, w = frame.shape[:2]
            canvas = frame.copy()
            poly = np.rint(np.array(cfg["zones"]["counter"]) * [w, h]).astype(np.int32)
            cv2.polylines(canvas, [poly], True, (0, 255, 255), 2)
            for obj in f["objects"]:
                b = [round(v * (w if j % 2 == 0 else h)) for j, v in enumerate(obj["bbox"])]
                cv2.rectangle(canvas, tuple(b[:2]), tuple(b[2:]), (0, 220, 0), 1)
                label = f"{obj['logical_object_id']} {obj['confidence']:.2f}"
                cv2.putText(
                    canvas,
                    label,
                    (b[0], max(12, b[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 220, 0),
                    1,
                )
            lines = [
                f"DIAGNOSTIC ONLY {f['timestamp_sec']:.2f}s",
                f"motion {f['counter_motion_fraction']:.3f}, "
                f"body gap {f['cashier_box_gap_widths']:.2f}",
            ]
            for e in variants["zone_motion_only"]["events"]:
                if 0 <= f["timestamp_sec"] - e["timestamp_sec"] <= 1.5:
                    lines.append(
                        e["type"]
                        .replace("CASH_", "")
                        .replace(
                            "PROBABLE_COUNTER_INTERACTION", "PROBABLE " + e.get("hypothesis", "")
                        )
                    )
            cv2.rectangle(canvas, (0, 0), (w, 24 + 20 * len(lines)), (24, 24, 24), -1)
            for j, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (5, 20 + 20 * j),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (240, 240, 240),
                    1,
                )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out / "diagnostic-debug.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    data["fps"] / data["stride"],
                    (w, h),
                )
                assert writer.isOpened()
            writer.write(canvas)
            if index in wanted:
                imwrite(out / "evidence" / f"{index:05d}-raw.jpg", frame)
                imwrite(out / "evidence" / f"{index:05d}-diagnostic.jpg", canvas)
                debug.set(cv2.CAP_PROP_POS_FRAMES, index // data["stride"])
                dok, dframe = debug.read()
                if dok:
                    imwrite(out / "evidence" / f"{index:05d}-existing-debug.jpg", dframe)
            index += 1
        cap.release()
        debug.release()
        if writer:
            writer.release()
        for mode, pred in variants.items():
            for event in pred["events"]:
                for ref in event["evidence_frames"]:
                    path = out / "evidence" / f"{ref['frame_index']:05d}-raw.jpg"
                    ref["image"] = str(path.relative_to(ROOT))
                    ref["image_sha256"] = sha(path)
                    ref["diagnostic_image"] = str(
                        path.with_name(path.name.replace("-raw", "-diagnostic")).relative_to(ROOT)
                    )
                    dbg = path.with_name(path.name.replace("-raw", "-existing-debug"))
                    if dbg.exists():
                        ref["existing_debug_image"] = str(dbg.relative_to(ROOT))
                # A compact sequence for every hypothesis: actual sampled source frames.
                refs = sorted(event["evidence_frames"], key=lambda r: r["timestamp_sec"])
                indices = sorted({0, len(refs) // 2, len(refs) - 1})
                page = Image.new("RGB", (3 * 320, 320), "white")
                draw = ImageDraw.Draw(page)
                for j, n in enumerate(indices):
                    ref = refs[n]
                    im = (
                        Image.open(ROOT / ref["image"]).crop((15, 350, 290, 590)).resize((320, 280))
                    )
                    page.paste(im, (j * 320, 35))
                    draw.text(
                        (j * 320 + 5, 5), f"{ref['timestamp_sec']:.2f}s {ref['role']}", fill="black"
                    )
                ep = out / "evidence" / f"{mode}-{event['id']}.jpg"
                page.save(ep, quality=92)
                event["evidence_sheet"] = str(ep.relative_to(ROOT))
            write_json(out / f"{mode}-events.json", pred)
        raw = read_json(ROOT / source["cache"] / "detections.json")
        history = read_json(ROOT / source["cache"] / "track_history.json")
        original_events_path = ROOT / source["cache"] / "events.json"
        original_events = (
            read_json(original_events_path)["events"] if original_events_path.exists() else []
        )
        money_events = [e["type"] for e in original_events if e["type"].startswith("MONEY_")]
        summaries.append(
            {
                "video": source["id"],
                "sampled_frames": len(data["frames"]),
                "sample_fps": data["fps"] / data["stride"],
                "raw_cash_detections": sum(len(f["objects"]) for f in raw["frames"]),
                "retained_cash_detections": sum(len(f["objects"]) for f in data["frames"]),
                "logical_tracks": len({o["logical_object_id"] for o in history["observations"]}),
                "inside_counter_cash_detections": sum(
                    o["inside_counter"] for f in data["frames"] for o in f["objects"]
                ),
                "prior_money_events": money_events,
                "prior_pipeline_executed_for_this_task": False,
                "cash_weights_sha256": raw["cash_weights_sha256"],
            }
        )
    write_json(META / "input-audit.json", summaries)


def report():
    results = read_json(META / "results.json")["results"]
    sources = read_json(META / "sources.json")
    input_rows = read_json(META / "input-audit.json")
    rows = []
    marker_review = read_json(META / "marker-review.json")["videos"]
    for source in sources:
        for review in marker_review[source["id"]]["false_markers"]:
            prediction = read_json(DATA / source["id"] / "zone_motion_only-events.json")
            candidates = [e for e in prediction["events"] if e["type"] == review["type"]]
            event = min(
                candidates, key=lambda e: abs(e["timestamp_sec"] - review["timestamp_sec_rounded"])
            )
            picture = imread(ROOT / event["evidence_frames"][0]["image"])
            h, w = picture.shape[:2]
            for obj in event["observed_features"]["cash_detections"]:
                box = [round(v * (w if j % 2 == 0 else h)) for j, v in enumerate(obj["bbox"])]
                cv2.rectangle(picture, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 1)
            imwrite(
                DATA / f"{source['id']}-false-cash.jpg",
                cv2.resize(picture[380:460, 40:220], (900, 400), interpolation=cv2.INTER_NEAREST),
            )
    for s in sources:
        a = next(r for r in results if r["video"] == s["id"] and r["mode"] == "body_box_reach")
        b = next(r for r in results if r["video"] == s["id"] and r["mode"] == "zone_motion_only")
        rows.append(
            [
                s["id"].upper(),
                a["true_interactions"],
                f"{a['detected']}/{a['true_interactions']}",
                f"{b['detected']}/{b['true_interactions']}",
                b["missed"],
                b["false_events"],
                len(marker_review[s["id"]]["false_markers"]),
            ]
        )
    result_table = html_table(
        [
            "Видео",
            "Реальных действий",
            "Рамка кассира + движение",
            "Только движение зоны",
            "Пропущено (зона)",
            "Ложных PROBABLE (зона)",
            "Ложных вспомогательных маркеров",
        ],
        rows,
    )
    observations = []
    details = []
    for s in sources:
        pred = read_json(DATA / s["id"] / "zone_motion_only-events.json")
        for e in pred["events"]:
            observations.append(
                [
                    s["id"].upper(),
                    f"{e['timestamp_sec']:.2f}",
                    e["type"],
                    e.get("hypothesis", "—"),
                    e["confidence"],
                    len(e["evidence_frames"]),
                ]
            )
            link = f"{s['id']}/evidence/zone_motion_only-{e['id']}.jpg"
            description = escape(
                json.dumps(
                    {
                        "observed_features": e["observed_features"],
                        "uncertainty_reasons": e["uncertainty_reasons"],
                        "decision_available_at_sec": e.get("decision_available_at_sec"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            details.append(
                f"<details><summary>{s['id'].upper()} {e['timestamp_sec']:.2f}s "
                f"— {e['type']}</summary>"
                f'<img src="{link}" alt="Evidence frames"><pre>{description}</pre></details>'
            )
    event_table = html_table(
        [
            "Видео",
            "Секунда",
            "Диагностический маркер",
            "Гипотеза",
            "Confidence*",
            "Evidence frames",
        ],
        observations,
    )
    failures = html_table(
        ["Видео / действие", "Наблюдения", "Где остановилась гипотеза"],
        [
            [
                "X-3: помещение",
                "Подход 16.15; последняя near-находка 16.62; исчезновение 17.32",
                (
                    "В окне до 18.12 только один кадр движения ≥2% (17.09). Следующий — 18.26, "
                    "уже вне окна. Рамка кассира не охватывает руку."
                ),
            ],
            [
                "X-2: помещение",
                "Подход 13.59; исчезновение 14.29; движение 14.52 и 14.76",
                (
                    "PROBABLE 14.52 только по движению зоны. Автор движения автоматически не "
                    "подтверждён."
                ),
            ],
            [
                "X-1: помещение",
                "Подход 14.04; движение 14.27/14.50; исчезновение 14.74",
                (
                    "PROBABLE 14.27. Фрагменты разных logical IDs связаны как гипотеза; "
                    "production tracks не изменены."
                ),
            ],
            [
                "X-1: извлечение",
                "Повторная находка 20.12; два наблюдения у кассира; удаление от зоны 0.366 ширины",
                "Только один кадр движения проходит порог: 19.42. Двухкадрового подтверждения нет.",
            ],
            [
                "X-5: помещение",
                "Подход 25.24; движение 25.94/26.17; исчезновение 26.40",
                "PROBABLE 25.94.",
            ],
            [
                "X-5: извлечение",
                "Повторная находка 32.48; два наблюдения у кассира; удаление 0.514 ширины",
                (
                    "Пик движения 31.78 — 1.54% зоны, ниже фиксированных 2%; нет подтверждающих "
                    "кадров."
                ),
            ],
            [
                "X-3: ложное повторное появление",
                "18.73 CASH_REAPPEARED_NEAR_COUNTER",
                (
                    "В bbox находится пустая кисть кассира. Нет последующих двух наблюдений cash "
                    "у кассира; извлечение не создано."
                ),
            ],
        ],
    )
    protected = read_json(META / "protected-sha256.json")
    changed = [p for p, h in protected.items() if sha(ROOT / p) != h]
    assert not changed, changed
    audit = {
        "protected_files_verified": len(protected),
        "changed": changed,
        "model_training_runs": 0,
        "production_event_detector_runs": 0,
        "rule_engine_runs": 0,
        "branch": "feat/counter-sequence-diagnostic",
        "protocol_sha256": sha(META / "protocol.json"),
        "core_sha256": sha(ROOT / "experiments/counter_sequence.py"),
    }
    write_json(META / "audit.json", audit)
    total_neg = next(r["negative_duration_sec"] for r in results if r["video"] == "x5")
    source_table = html_table(
        [
            "Видео",
            "Кадров",
            "FPS анализа",
            "Raw cash",
            "После NMS",
            "Logical tracks",
            "Cash внутри зоны",
        ],
        [
            [
                r["video"],
                r["sampled_frames"],
                f"{r['sample_fps']:.2f}",
                r["raw_cash_detections"],
                r["retained_cash_detections"],
                r["logical_tracks"],
                r["inside_counter_cash_detections"],
            ]
            for r in input_rows
        ],
    )
    page = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Диагностика взаимодействия со счётчиком</title>
<style>body{{font:16px/1.5 system-ui;
max-width:1250px;
margin:32px auto;
padding:0 20px;
color:#18232c}}
table{{border-collapse:collapse;
width:100%;
font-size:13px;
margin:18px 0}}th,td{{border:1px solid #ccd4dd;
padding:9px;
text-align:left}}
th{{background:#edf2f6}}h2{{margin-top:35px}}pre{{white-space:pre-wrap;
font-size:12px;
background:#f4f6f8;
padding:15px}}
img,video{{max-width:100%}}details{{margin:12px 0}}.notice{{background:#fff3dd;
padding:16px;
border-left:5px solid #b77700}}</style>
<h1>Взаимодействие кассира со счётчиком: диагностический эксперимент</h1>
<p>18 октября 2026 · отдельная ветка <code>feat/counter-sequence-diagnostic</code></p>
<p class="notice"><b>Частичное восстановление возможно без детекций денег внутри аппарата.</b>
С более строгим признаком рамки кассира найдено 2/6 реальных действий; с движением зоны — 3/6.
У обоих вариантов 0 ложных PROBABLE на этих четырёх роликах. Однако обнаружение извлечения
не прошло проверку ни в X-1, ни в X-5, а в X-3 появился ложный диагностический маркер повторного
появления cash.
Результат не готов к production.</p>
<h2>Результаты по видео</h2>{result_table}
<p>Единица оценки — одно видимое помещение или одно извлечение, а не отдельный кадр и не весь цикл.
Precision PROBABLE на этой выборке: 2/2 и 3/3; recall: 2/6 (33.3%) и 3/6 (50%).
Это маленький диагностический replay знакомых видео, не независимый тест обобщения.
X-1 и X-5 присутствовали в данных прежнего YOLO. Новое обучение не выполнялось.</p>
<h2>Что использовано</h2>{source_table}
<p>Для X-3/X-2/X-1 прочитаны сохранённые detections, track_history, events, timeline и кадры
debug.mp4
основной модели из <code>data/events/audit/after/</code>. Исходные debug-видео также доступны в
<code>data/events/x*/</code>. Кэш отклонённого counter-repair кандидата просмотрен для аудита, но
его
находки не подмешивались в основной эксперимент. В исходных events нет денежных событий.</p>
<p>Для X-5 отсутствовал полный кэш: выполнен inference той же cash-модели и person-модели на RTX
3050,
с теми же порогами, а также вызваны неизменённые role/object trackers. Production Event Detector и
R01–R04
не запускались. Совпадение SHA-256 денежных весов проверено. Вычисление простой разности кадров идёт
на CPU.</p>
<h2>Логика эксперимента</h2>
<ol><li>Выбираются cash-находки вне COUNTER_ZONE с исходным confidence ≥0.25.</li>
<li>Близость к кассиру берётся из геометрии рамок людей. Для подхода нужны ≥2 таких наблюдения,
сокращение расстояния минимум на 0.25 ширины счётчика и финальное расстояние ≤0.75 ширины за 3
секунды.</li>
<li>Разные logical tracks допускаются только как временная гипотеза: разрыв ≤1.5 с и перемещение ≤1
ширины.
Их исходные IDs остаются в evidence. Это не подтверждённое восстановление идентичности.</li>
<li>Исчезновение означает отсутствие подходящего внешнего продолжения ≥0.5 с после последней
near-находки.
Оно сохраняется отдельным маркером и само по себе не порождает взаимодействие.</li>
<li>Движение внутри исходного полигона: разность размытых grayscale кадров, компенсация общего
сдвига
яркости, delta ≥18/255, morphology 3×3; изменено ≥2% площади. Нужно ≥2 кадра.</li>
<li>Вариант body_box_reach дополнительно требует рамку кассира в пределах 0.35 ширины от аппарата.
Вариант zone_motion_only требует только видимого кассира: автор движения остаётся неизвестным.
Численные пороги между вариантами не менялись. Второй вариант добавлен после визуального выявления
того, что person bbox не охватывает вытянутую руку X-3, до оценки гипотез.</li>
<li>Для возможного помещения объединяются подход, исчезновение и движение в ограниченном временном
окне.
Для возможного извлечения дополнительно нужны предыдущая гипотеза помещения, повторное появление
после паузы, новое движение у аппарата, ≥2 последующих находки около кассира и удаление от
зоны.</li>
<li>После повторного появления отдельно сохраняется движение cash к клиенту. Оно не доказывает
передачу,
разделение пачки или сумму денег.</li></ol>
<p>Имена видео и размеченные таймкоды не доступны функции <code>analyze(frames, parameters)</code>.
Ground truth загружается только после получения предсказаний. В отчёте таймкоды — ссылки для
проверки.
Все детекции с центром внутри счётчика исключены из денежных последовательностей; детекции купюр
внутри
аппарата для этих результатов не требуются.</p>
<h2>Все диагностические маркеры варианта zone_motion_only</h2>{event_table}
<p>* Confidence — прозрачная эвристическая оценка полноты evidence, не калиброванная вероятность:
подход 0.60, исчезновение 0.40, повторное появление 0.45; PROBABLE с рамкой 0.65–0.70,
только по движению зоны ≤0.55. Эти числа не проходят в production-пороги.
Один эпизод может иметь несколько маркеров; они не суммируются как несколько реальных
взаимодействий.</p>
<h2>Таблица ошибок и остановок</h2>{failures}
<img src="x3-false-cash.jpg" alt="Ложный cash: красная рамка на пустой кисти; деньги остаются в
аппарате">
<p>Красная рамка cash охватывает пустую кисть. Разбор вспомогательных маркеров
сохранён отдельно в marker-review.json.</p>
<p>В X-1 после 20.12 модель снова видит cash; есть два наблюдения рядом с кассиром и позднее
находки с ассоциацией customer в 22.22/23.39. Это поддерживает последовательность «снова cash у
кассира →
движение к клиенту», но не доказывает извлечение, идентичность пачки или её разделение.</p>
<h2>Отрицательные примеры X-5</h2>
<p>Просмотрены кадры всего ролика с шагом 0.5 с, плюс более подробные кадры loading/unloading и
debug.
Отрицательные интервалы: 0–25.3, 28–31.3, 33–71.6 с, всего {total_neg:.1f} с.
Они включают ID/кошелёк, деньги в руках вдали от аппарата, ожидание, уход/приход клиента и новый ID.
В этих интервалах 0 ложных PROBABLE у обоих вариантов. X-5 целиком не объявлен отрицательным:
в нём есть одно реальное помещение и одно извлечение. Маркер APPROACHED в 25.24 — предварительный
подход; он не засчитан как взаимодействие и показан отдельно.</p>
<p>Трудных отрицательных примеров «кассир касается пустого аппарата без денег» и «перекладывает
предмет
вплотную к счётчику» здесь недостаточно. Поэтому 0 ложных событий на 67.2 с не означает
установленную
надёжность. Разметка интервалов выполнена ассистентом и требует проверки человеком.</p>
<h2>Evidence каждого предположения</h2>{"".join(details)}
<h2>Минимальный следующий шаг</h2>
<p>Проверить отдельный признак кисти/запястья кассира: последовательность «кисть приблизилась к зоне
→
контакт/перекрытие → кисть ушла», включая нижний приёмник. Сначала разметить и проверить несколько
положительных и трудных отрицательных эпизодов; можно испытать готовую pose/hand-модель без
дообучения.
Для проверки самого принципа допустима временная ручная разметка кистей только как отдельный
диагностический
канал, без подмешивания её в итоговую автоматическую метрику. Не снижать пороги ради знакомых
таймкодов.</p>
<p>Сохранять PROBABLE как дополнительное наблюдение с unknown actor/identity при нехватке evidence.
Текущие MONEY_TO_COUNTER/MONEY_FROM_COUNTER и R01–R04 не изменены. Пропавшая detection и одиночное
движение зоны не должны превращаться в подтверждённое бизнес-событие.</p>
<h2>Файлы и проверки</h2><p><code>datasets/counter_sequence_v1/</code>: protocol, sources,
ground-truth,
results, input-audit и audit. <code>data/counter_sequence_v1/x*/</code>: features, два events.json,
evidence и diagnostic-debug.mp4. У каждого события сохранены timestamp, признаки, confidence,
ссылки/хеши кадров и причины неопределённости. Проверены {len(protected)} защищённых исходных файлов
— изменений нет.</p>
<pre>./.venv/Scripts/python.exe -m experiments.counter_sequence_run
./.venv/Scripts/python.exe -m experiments.counter_sequence_evidence</pre>
<p>Первый вызов пересчитывает только диагностические гипотезы из сохранённых features. Второй
экспортирует
evidence и отчёт. Обучение и production pipeline не запускаются.</p></html>"""
    (DATA / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    export()
    report()
