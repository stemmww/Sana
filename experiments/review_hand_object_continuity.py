"""Post-prediction evidence report. Does not modify detections or action decisions."""

import html
import json
from collections import Counter

import cv2
import numpy as np

from experiments.counter_occupancy import ROOT, imwrite
from experiments.counter_occupancy import read_json as read
from experiments.counter_occupancy import write_json as write
from experiments.hand_object_continuity import palm_frames
from experiments.run_hand_object_continuity import META, OUT


def frame_at(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    if not ok:
        raise ValueError(f"Evidence frame missing: {index}")
    return image


def build():
    evaluation = read(OUT / "evaluation.json")
    summary = read(OUT / "summary.json")
    document = """<!doctype html><meta charset="utf-8"><title>Sana: R04 hand continuity</title>
<style>body{font:16px system-ui;max-width:1150px;margin:35px auto;background:#f5f6f8;color:#182333}
table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccc;padding:10px}
img{max-width:100%}pre{white-space:pre-wrap;background:white;padding:15px}details{margin:12px 0}
.warning{background:#fff0da;padding:20px}</style>
<h1>R04: область кисти и кратковременная непрерывность изображения</h1>
<p class="warning">Новых подтверждённых событий и нарушений: 0. Кандидат отклонён для интеграции.
Проекция кисти улучшила одно наблюдение получателя в X-1, но потеряла гипотезу извлечения X-5.
Правила, API, веса и production tracking не изменены.</p>
<p>Wrist — прежняя диагностика; palm — геометрическая проекция кисти;
association_only — проекция только для привязки денег, контакты по исходным запястьям.
Все результаты ниже — гипотезы, не semantic events. Три варианта используют одинаковые пороги.
Это повторно изученные диагностические видео, не независимый тест.</p>
<table><tr><th>Видео</th><th>Вариант</th><th>Загрузка</th><th>Извлечение</th><th>Возврат</th>
<th>Гипотезы вне целевых интервалов</th></tr>"""
    kinds = ["CASH_LOADING_ACTION", "CASH_EXTRACTION_ACTION", "CASH_HANDOVER_TO_CUSTOMER"]
    for row in evaluation["rows"]:
        counts = []
        for kind in kinds:
            targets = [t for t in row["targets"] if t["type"] == kind]
            counts.append(
                f"{sum(bool(t['hits']) for t in targets)}/{len(targets)}" if targets else "—"
            )
        document += (
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(x))}</td>"
                for x in [row["video"], row["variant"], *counts, len(row["unmatched_hypotheses"])]
            )
            + "</tr>"
        )
    document += "</table><p>На пяти проверенных отрицательных фрагментах новых гипотез нет. "
    document += "Пустое касание нижнего лотка отсутствует; специфичность не установлена.</p>"
    document += '<p><a href="evaluation.json">Оценка</a> | <a href="protocol.json">Протокол</a> | '
    document += '<a href="summary.json">Время и защита production</a> | '
    document += '<a href="regression.xml">Тесты</a></p>'
    sources = read(META / "sources.json")
    for source in sources:
        name = source["id"]
        output = OUT / name
        hand_frames = read(ROOT / "data/counter_hand_v1" / name / "hands.json")["frames"]
        hands = palm_frames(hand_frames)
        changes = read(output / "association-changes.json")
        flow = read(output / "texture-tracks.json")
        cap = cv2.VideoCapture(source["video"])
        row = next(r for r in summary["videos"] if r["video"] == name)
        document += f"<h2>{name.upper()}</h2><pre>{html.escape(json.dumps(row, indent=2))}</pre>"
        stops = dict(Counter(t["stop_reason"] for t in flow["tracks"]))
        document += f"<p>Причины остановки optical flow: {html.escape(str(stops))}</p>"
        document += f'<p><a href="{name}/actions.json">Все action-кандидаты</a> | '
        document += f'<a href="{name}/association-only.json">Ablation</a> | '
        document += f'<a href="{name}/texture-tracks.json">Все texture tracks</a></p>'
        document += (
            "<h3>Изменения привязки денег</h3><p>Слева оригинал, справа геометрические наблюдения. "
        )
        document += "Круг — измеренное запястье; крест — расчётный центр кисти; "
        document += "жёлтая рамка — реальная YOLO detection.</p>"
        for number, change in enumerate(changes):
            before, after = change["before"], change["after"]
            index = before["frame_index"]
            img = frame_at(cap, index)
            painted = img.copy()
            hf = min(hands, key=lambda f: abs(f["timestamp_sec"] - before["timestamp_sec"]))
            for hand in hf["hands"]:
                if hand["owner"] not in {"cashier", "customer"}:
                    continue
                color = (40, 180, 0) if hand["owner"] == "cashier" else (0, 0, 220)
                wrist, palm = [
                    tuple(np.rint(hand[k]).astype(int)) for k in ["observed_wrist_xy", "xy"]
                ]
                cv2.line(painted, tuple(np.rint(hand["elbow"][:2]).astype(int)), wrist, color, 1)
                cv2.circle(painted, wrist, 3, color, 1)
                cv2.line(painted, wrist, palm, color, 1)
                cv2.drawMarker(painted, palm, color, cv2.MARKER_CROSS, 8, 1)
            box = np.rint(np.asarray(before["bbox"]) * ([img.shape[1], img.shape[0]] * 2)).astype(
                int
            )
            cv2.rectangle(painted, tuple(box[:2]), tuple(box[2:]), (0, 220, 220), 1)
            # Review crop is derived from the observed cash and all known hands.
            points = [box[:2], box[2:]] + [
                h["xy"] for h in hf["hands"] if h["owner"] in {"cashier", "customer"}
            ]
            low = np.maximum(np.floor(np.min(points, axis=0) - 35).astype(int), [0, 0])
            high = np.minimum(
                np.ceil(np.max(points, axis=0) + 35).astype(int), [img.shape[1], img.shape[0]]
            )
            crops = [x[low[1] : high[1], low[0] : high[0]] for x in [img, painted]]
            panel = cv2.resize(np.hstack(crops), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            relative = f"{name}/association-review-{number + 1}.jpg"
            imwrite(OUT / relative, panel)
            change["evidence_frames"] = [
                {"frame_index": index, "timestamp_sec": before["timestamp_sec"], "path": relative}
            ]
            document += (
                f"<details><summary>{before['timestamp_sec']:.3f}s {before['logical_cash_id']}: "
            )
            document += (
                f"{before['owner']} → {after['owner']}</summary><img src='{relative}'></details>"
            )
        write(output / "association-review.json", changes)
        document += "<h3>Проверка текстуры</h3>"
        for number, track in enumerate(flow["tracks"]):
            chosen = [(track["seed_frame"], None)]
            if track["samples"]:
                last = track["samples"][-1]
                chosen.append((last["frame_index"], last))
            paths = []
            for index, node in chosen:
                img = frame_at(cap, index)
                if node:
                    bbox = node["bbox_px"]
                else:
                    bbox = np.asarray(track["seed_owner"]["bbox"]) * (
                        [img.shape[1], img.shape[0]] * 2
                    )
                a, b, c, d = np.rint(bbox).astype(int)
                cv2.rectangle(img, (a, b), (c, d), (0, 200, 255), 1)
                relative = f"{name}/texture-evidence/{number + 1}-{index}.jpg"
                imwrite(OUT / relative, img)
                paths.append(
                    {
                        "frame_index": index,
                        "path": relative,
                        "bbox_basis": "texture_estimate" if node else "observed_cash_detection",
                    }
                )
            track["evidence_frames"] = paths
        write(output / "texture-review.json", flow)
        document += f'<p><a href="{name}/texture-review.json">Кадры траекторий</a></p>'
        cap.release()
    document += "<h2>Следующий шаг</h2><p>Сохранить исходные запястья для контактов. Не заменять "
    document += "владельца по проекции кисти автоматически. Нужны новые видео передачи денег, "
    document += "ID и пустых рук: объект виден до и после передачи. Для загрузки проверить кисть "
    document += (
        "и пальцы, а не увеличивать зоны. Локальный detector кисти пригоден для проверки лишь при "
    )
    document += "наличии подходящих весов; в этом эксперименте новые модели не запускались.</p>"
    (OUT / "report.html").write_text(document, encoding="utf-8")
    print(OUT / "report.html")


if __name__ == "__main__":
    build()
