"""Audit preservation and export a reviewable HTML report for the wrist diagnostic."""
# Long literal HTML rows are kept together for review.
# ruff: noqa: E501

from __future__ import annotations

from collections import Counter

from experiments.counter_hand import analyze
from experiments.counter_hand_data import DATA, META
from experiments.counter_hand_evidence import annotate
from experiments.counter_occupancy import ROOT, imread, imwrite, read_json, sha, write_json


def main():
    p = read_json(META / "protocol.json")
    results = read_json(META / "results.json")["results"]
    protected = read_json(META / "protected-sha256.json")
    changed = [
        name
        for name, digest in protected.items()
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest
    ]
    assert not changed, changed
    audited, artifact_refs = [], []
    # Formatting-only edits must preserve decisions; update source hashes only after equality.
    for source in read_json(META / "sources.json"):
        out = DATA / source["id"]
        frames = read_json(out / "hands.json")["frames"]
        cash = read_json(ROOT / "data/counter_sequence_v1" / source["id"] / "features.json")
        old = read_json(out / "predictions.json")
        recomputed = analyze(frames, cash["frames"], p)
        for key in ["contacts", "markers", "summary"]:
            assert recomputed[key] == old[key], (source["id"], key)
        evidence = read_json(out / "predictions-with-evidence.json")
        timeline = [
            "Diagnostic observations only; no production events. Scores are not probabilities."
        ]
        timeline += [
            f"{m['timestamp_sec']:.3f}s {m['type']} owner={m['owner']} "
            f"confidence={m['confidence']:.2f} contact={m['contact_id']}"
            for m in evidence["markers"]
        ]
        for contact in evidence["contacts"]:
            if contact["withdrawal_hypothesis"]:
                cash_times = sorted(
                    {
                        n["timestamp_sec"]
                        for n in contact["cash_links"]
                        if n["wrist_owner"] == "cashier"
                    }
                )
                available = cash_times[p["cash_min_samples"] - 1]
                contact["hypothesis_available_sec"] = available
                timeline.append(
                    f"{available:.3f}s WITHDRAWAL_HYPOTHESIS contact={contact['id']} "
                    f"confidence={contact['confidence']:.2f}; first cash={cash_times[0]:.3f}s"
                )
        if not evidence["contacts"]:
            timeline.append("No sustained lower-output visits or withdrawal hypotheses.")
        (out / "timeline.txt").write_text("\n".join(timeline) + "\n", encoding="utf-8")
        for path, doc in [
            (out / "predictions.json", old),
            (out / "predictions-with-evidence.json", evidence),
        ]:
            doc["decision_code_sha256"] = sha(ROOT / "experiments/counter_hand.py")
            write_json(path, doc)
        all_hands = [h for f in frames for h in f["hands"]]
        near = [h for h in all_hands if h["distance_widths"] <= 0.75]
        inside = [h for h in all_hands if h["inside_output_zone"]]
        stats = {
            "video": source["id"],
            "near_output_owner_observations": dict(Counter(h["owner"] for h in near)),
            "inside_owner_observations": dict(Counter(h["owner"] for h in inside)),
            "unknown_reasons": dict(
                Counter(r for h in all_hands if h["owner"] == "unknown" for r in h["uncertainty"])
            ),
        }
        audited.append(stats)
        for contact in evidence["contacts"]:
            for node in contact["evidence_frames"]:
                artifact_refs.extend([node["raw"], node["diagnostic"]])
        for marker in evidence["markers"]:
            artifact_refs.extend(marker["evidence_frames"])
    assert all((ROOT / ref).is_file() for ref in artifact_refs)
    write_json(
        META / "audit.json",
        {
            "protected_files": len(protected),
            "changed": changed,
            "verified_evidence_references": len(artifact_refs),
            "decision_equivalence_after_formatting": True,
            "protocol_sha256": sha(META / "protocol.json"),
            "ownership_detail": audited,
        },
    )
    calibration = imread(DATA / "calibration-raw.jpg")
    imwrite(DATA / "calibration-zones.jpg", annotate(calibration, [], p))
    rows, ownership_rows, contacts_html, artifact_links = [], [], [], []
    for row in results:
        v = row["video"]
        rows.append(
            f"<tr><td>{v.upper()}</td><td>{row['real_withdrawals']}</td><td>{row['detected']}</td><td>{row['missed']}</td><td>{row['false_withdrawals']}</td><td>{row['cash_linked_contacts']}</td><td>{row['unknown_owner_contacts']}</td></tr>"
        )
        unknown_near = next(a for a in audited if a["video"] == v)[
            "near_output_owner_observations"
        ].get("unknown", 0)
        ownership_rows.append(
            f"<tr><td>{v.upper()}</td><td>{row['unknown_wrist_observations']} / {row['wrist_observations']}</td><td>{unknown_near}</td><td>{row['unknown_owner_contacts']} / {row['output_contacts']}</td></tr>"
        )
        predicted = read_json(DATA / v / "predictions-with-evidence.json")
        for c in predicted["contacts"]:
            first_cash = c["cash_links"][0]["timestamp_sec"] if c["cash_links"] else None
            approach = c["wrist_evidence"][0]["timestamp_sec"]
            contacts_html.append(
                f'<section><h3>{v.upper()}: {approach:.2f} → {c["timestamp_sec"]:.2f} → {c["leave_sec"]:.2f} с</h3><p>Подход → вход в зону → отход. Первые деньги после отхода: {first_cash:.2f} с. В зоне: {c["observed_features"]["inside_samples"]} кадров, {c["observed_features"]["inside_duration_sec"]:.2f} с. Владелец: {c["owner"]}. Оценка свидетельств: {c["confidence"]:.2f} (не вероятность).</p><a href="{v}/contact-1.jpg"><img src="{v}/contact-1.jpg" alt="Кадры взаимодействия {v}"></a><p><a href="{v}/predictions-with-evidence.json">Все признаки, неопределённости, cash IDs и evidence frames</a> · <a href="{v}/wrist-debug.mp4">Диагностическое видео</a></p></section>'
            )
        artifact_links.append(
            f'<li>{v.upper()}: <a href="{v}/poses.json">Pose</a> · <a href="{v}/hands.json">Координаты и принадлежность кистей</a> · <a href="{v}/predictions-with-evidence.json">Последовательности и evidence</a> · <a href="{v}/wrist-debug.mp4">Debug MP4</a></li>'
        )
    report = (
        """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Диагностика запястий у счётчика</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 22px;color:#17212b;background:#fafbfc}h1,h2,h3{line-height:1.25}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{border:1px solid #d4dce4;padding:10px;text-align:left}th{background:#edf2f7}.notice{padding:18px;border-left:5px solid #d48c00;background:#fff4d8}.good{padding:18px;border-left:5px solid #2d8868;background:#e8f5ef}a{color:#075aab}section{margin:28px 0}img{max-width:100%;height:auto}code,pre{background:#eaf0f6}pre{padding:16px;overflow:auto}.zones{max-width:330px}</style>
<h1>Кисть кассира и нижний приёмник</h1><p>Независимый диагностический эксперимент. YOLO11s Pose, официальные предобученные веса, RTX 3050 Laptop, imgsz=1280, около 15 кадров/с. Обучение не проводилось.</p>
<h2>Результаты</h2><table><thead><tr><th>Видео</th><th>Реальных извлечений</th><th>Найдено</th><th>Пропущено</th><th>Ложных</th><th>Связано с деньгами</th><th>Unknown у приёмника</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table>
<p class="good"><b>На этих четырёх роликах: найдено 2/2 извлечения, пропусков 0, ложных гипотез 0.</b> Предыдущий анализ движения и bbox человека находил 0/2 извлечений. X‑3 и X‑2 заканчиваются без извлечения денег — отсутствие гипотез там правильно.</p>
<p class="notice">Два положительных эпизода одной камеры — ограниченная проверка. Реальных пустых касаний нижнего приёмника в доступных примерах нет; пользователь подтвердил отсутствие отдельного ролика. Качество на этом важном отрицательном типе действий <b>не измерено</b>. Это не разрешение на production-интеграцию.</p>
<h2>Ошибки и ограничения</h2><table><tr><th>Проверка</th><th>Результат / ограничение</th></tr>
<tr><td>Пропущенные и ложные извлечения</td><td>0 и 0 на двух размеченных извлечениях и просмотренных отрицательных фрагментах. Разметка — визуальный просмотр ассистентом, без независимой человеческой проверки.</td></tr>
<tr><td>Пустые касания нижнего приёмника</td><td>0 реальных примеров. Есть только синтетический тест запрета «касание без денег → извлечение»; он не измеряет качество модели на видео.</td></tr>
<tr><td>Загрузка сверху</td><td>4 эпизода, ни один не принят за извлечение снизу. Это взаимодействия без извлечения, а не пустые касания.</td></tr>
<tr><td>Руки у стола, ID, деньги вне аппарата</td><td>Нет дополнительных гипотез. В X‑5 просмотрены 67,2 с отрицательных интервалов: работа руками, ожидание, смена клиента и ID в конце.</td></tr>
<tr><td>Ошибочный cash в X‑3 около 18,73 с</td><td>Старый детектор принял пустую поднятую руку за деньги. Новая цепочка нижнего приёмника отсутствует, ложного извлечения не возникло.</td></tr>
<tr><td>Точность контакта</td><td>Запястье не входит в узкий физический контур приёмника даже при двух реальных извлечениях. Использован заранее выделенный допуск для кисти. Нахождение в зоне означает близость в 2D, а не доказанный физический контакт.</td></tr>
<tr><td>Принадлежность и tracking</td><td>Много коротких неизвестных треков в остальных частях кадра, включая отражения. Они сохранены как unknown. У обоих контактов есть 8 наблюдений руки кассира. Точность всех keypoints и всех присвоений роли не размечена.</td></tr>
<tr><td>Причина улучшения</td><td>Одновременно применены запястья и более частая выборка. Отдельный вклад этих двух изменений не измерен.</td></tr></table>
<h2>Принадлежность руки: точные знаменатели</h2><table><tr><th>Видео</th><th>Unknown / все наблюдения запястий</th><th>Unknown вблизи зоны*</th><th>Unknown / эпизоды внутри зоны</th></tr>"""
        + "".join(ownership_rows)
        + """</table><p>*В пределах 0,75 ширины счётчика от зоны. Это покадровые наблюдения, повторяющиеся для одной руки; они не являются числом людей или событий. Внутри зоны приёмника неизвестных наблюдений нет.</p>
<h2>Что именно проверялось</h2><p>Существующая COUNTER_ZONE скопирована без изменений. Физический контур нижнего приёмника и COUNTER_OUTPUT_ZONE с допуском для кисти заданы по статическому пустому кадру X‑4 до запуска модели, одинаково для всех четырёх видео. Жёлтый — исходный счётчик, голубой — видимый нижний приёмник, зелёный — зона наблюдения запястья.</p><a href="calibration-zones.jpg"><img class="zones" src="calibration-zones.jpg" alt="Зоны счётчика"></a>
<p>Поза сопоставляется с существующей геометрической ролью по torso и IoU ≥ 0,30, при отрыве от конкурирующего совпадения ≥ 0,15. Для руки нужны плечо, локоть и запястье с confidence ≥ 0,35 и три последовательных согласованных наблюдения. Руку разрешено видеть за границами bbox человека. Facial keypoints отброшены; распознавания лиц и биометрической идентификации нет.</p>
<p>Последовательность: наблюдаемое приближение → минимум два кадра в выходной зоне → наблюдаемый отход. Потеря детекции не заменяет отход. Для гипотезы извлечения дополнительно нужны минимум две cash-детекции у этой руки в течение 2 с после отхода и отсутствие уже видимых у неё денег непосредственно перед контактом. Cash track ID может меняться. Confidence — эвристическая оценка свидетельств, не вероятность.</p>
<p>Ядро решения принимает только наблюдения, относительное время и геометрию. Имена роликов и размеченные интервалы используются при загрузке файлов и оценке после предсказания. Порогов по именам видео или известным таймкодам нет. Параметр cash_new_min_gap_sec в замороженном протоколе не используется; фактическая проверка новизны — отсутствие совпадающего cash за 0,5 с перед входом.</p>
<h2>Кадры найденных последовательностей</h2>"""
        + "".join(contacts_html)
        + """
<h2>Отрицательные примеры</h2><ul><li><a href="x3/review-3.jpg">X‑3: загрузка сверху, отход пустой руки, деньги остаются в аппарате</a></li><li><a href="x2/review-2.jpg">X‑2: деньги в руках и загрузка сверху</a></li><li><a href="x5/review-4.jpg">X‑5: движение у стола, загрузка сверху, ожидание</a></li><li><a href="x5/review-11.jpg">X‑5: новый клиент и ID вдали от нижнего приёмника</a></li></ul>
<h2>Минимальный следующий шаг</h2><p>Собрать короткие ролики той же камеры: касание пустого нижнего приёмника, нажатие кнопки, движение у стола рядом с ним, касание с уже находящимися в руке деньгами или ID. Проверить на них этот зафиксированный алгоритм. После проверки качества можно добавить отдельный источник наблюдений hand_counter_sequence с owner, cash-after-leave и evidence. Event Detector должен сам проверять денежный контекст; касание или исчезновение кисти не должны напрямую создавать MONEY_FROM_COUNTER.</p>
<h2>Артефакты и воспроизведение</h2><ul>"""
        + "".join(artifact_links)
        + f"""</ul><p>Проверены SHA‑256 для {len(protected)} существующих файлов: production-код, конфигурация, веса, прежние результаты и код предыдущего эксперимента сохранены. Ветвь: feat/counter-wrist-diagnostic. Полный pipeline и R01–R04 не запускались.</p><p><a href="../../datasets/counter_hand_v1/protocol.json">Протокол</a> · <a href="../../datasets/counter_hand_v1/results.json">Таблица JSON</a> · <a href="../../datasets/counter_hand_v1/review.json">Визуальная проверка и ограничения</a> · <a href="../../datasets/counter_hand_v1/audit.json">Аудит</a> · <a href="https://docs.ultralytics.com/tasks/pose/">Документация YOLO Pose</a></p>
<pre>.\\.venv\\Scripts\\python.exe -m experiments.counter_hand_data infer
.\\.venv\\Scripts\\python.exe -m experiments.counter_hand_run
.\\.venv\\Scripts\\python.exe -m experiments.counter_hand_evidence
.\\.venv\\Scripts\\python.exe -m experiments.counter_hand_report</pre><p>infer использует сохранённые pose-кэши при их наличии. prepare повторно не запускается: протокол уже зафиксирован. Артефакты эксперимента локальны, автоматической интеграции нет.</p></html>"""
    )
    (DATA / "report.html").write_text(report, encoding="utf-8")
    print(
        "Report ready; protected",
        len(protected),
        "evidence references",
        len(artifact_refs),
        "unchanged decisions",
    )


if __name__ == "__main__":
    main()
