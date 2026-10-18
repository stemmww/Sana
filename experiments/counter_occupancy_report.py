"""Render the independent experiment's saved results, without fitting any model."""

from html import escape
from pathlib import Path

from PIL import Image, ImageDraw

from experiments.counter_occupancy import DATA, META, ROOT, read_json, sha, sheet, write_json


def percent(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
        )
        + "</tbody></table>"
    )


def main():
    opencv = read_json(META / "opencv-validation-results.json")
    cnn = read_json(META / "cnn-validation-results.json")
    cv_selected = next(e for e in opencv["evaluations"] if e["model"] == opencv["selected_model"])
    nn_selected = next(e for e in cnn["evaluations"] if e["model"] == cnn["selected_model"])
    evaluations = opencv["evaluations"] + cnn["evaluations"]
    dev_cv = read_json(META / "opencv-dev-results.json")
    dev_nn = read_json(META / "cnn-dev-results.json")
    dev = {m["name"]: m for m in dev_cv["models"] + dev_nn["models"]}
    manifest = read_json(META / "annotations.json")
    val = read_json(META / "validation.json")["items"]
    cv_rows = {r["id"]: r for r in cv_selected["rows"]}
    nn_rows = {r["id"]: r for r in nn_selected["rows"]}
    chosen_names = {cv_selected["model"], nn_selected["model"]}
    summary = []
    for e in evaluations:
        six = e["metrics"]["six_counter_without_hand"]
        empty = e["metrics"]["empty_counter_28"]
        whole = e["metrics"]["all_clear"]
        summary.append(
            {
                "model": e["model"],
                "selected_on_dev": e["model"] in chosen_names,
                "counter_cash_detected": six["tp"],
                "counter_cash_total": 6,
                "false_cash_empty_counter": empty["fp"],
                "empty_total": 28,
                "all_clear_precision": whole["precision"],
                "all_clear_recall": whole["recall"],
                "all_clear_balanced_accuracy": whole["balanced_accuracy"],
                "threshold": e["threshold"],
            }
        )
    errors = [
        {"model": e["model"], **r}
        for e in evaluations
        for r in e["rows"]
        if r["result"] in {"FN", "FP"}
    ]
    write_json(
        META / "comparison.json",
        {
            "decision": "Do not integrate: required positive recall insufficient.",
            "fixed_validation_frames": 42,
            "scored_clear_frames": 36,
            "uncertain_unscored_frames": 6,
            "annotation_provenance": (
                "Provisional assistant visual review, not independent human ground truth."
            ),
            "selected_opencv": cv_selected["model"],
            "selected_cnn": nn_selected["model"],
            "models": summary,
        },
    )
    write_json(META / "errors.json", errors)

    # Eight enlarged original crops: six required positives and two representative negatives.
    examples = [i for i in val if i["subset"] == "six_counter_without_hand"]
    examples += [i for i in val if i["id"] in {"v03_00600", "v03_01600"}]
    canvas = Image.new("RGB", (4 * 300, 2 * 420), "#f1f3f5")
    draw = ImageDraw.Draw(canvas)
    for n, item in enumerate(examples):
        x, y = (n % 4) * 300, (n // 4) * 420
        cr, nr = cv_rows[item["id"]], nn_rows[item["id"]]
        draw.text((x + 8, y + 8), f"{item['id']} GT: {item['label']}", fill="black")
        draw.text(
            (x + 8, y + 27),
            f"OpenCV: {cr['prediction']} [{cr['result']}]",
            fill="#13733b" if cr["result"] == "OK" else "#b02020",
        )
        draw.text(
            (x + 8, y + 46),
            f"MobileNet: {nr['prediction']} [{nr['result']}]",
            fill="#13733b" if nr["result"] == "OK" else "#b02020",
        )
        crop = Image.open(ROOT / item["crop"]).convert("RGB")
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
        canvas.paste(crop, (x + 8, y + 72))
        draw.text(
            (x + 8, y + 385), f"CV score {cr['score']:.3f} / NN {nr['score']:.3f}", fill="black"
        )
    canvas.save(DATA / "validation-examples.jpg", quality=95)
    training_examples = [
        i
        for i in manifest["items"]
        if i["selected"] and i["split"] == "train" and i["label"] == "CASH_PRESENT"
    ]
    training_examples += [i for i in manifest["items"] if i["id"] in {"c01_00000", "c04_06500"}]
    sheet(training_examples, DATA / "training-examples.jpg")

    main_rows = []
    for s in summary:
        if s["model"] not in chosen_names | {"absolute_difference", "compensated_difference_p98"}:
            continue
        main_rows.append(
            [
                s["model"],
                f"{s['counter_cash_detected']}/6",
                f"{s['false_cash_empty_counter']}/28",
                percent(s["all_clear_precision"]),
                percent(s["all_clear_recall"]),
                percent(s["all_clear_balanced_accuracy"]),
            ]
        )
    six_rows = []
    for i in val:
        if i["subset"] != "six_counter_without_hand":
            continue
        cr, nr = cv_rows[i["id"]], nn_rows[i["id"]]
        six_rows.append(
            [
                i["id"],
                "CASH_PRESENT",
                cr["prediction"],
                f"{cr['score']:.4f}",
                nr["prediction"],
                f"{nr['score']:.4f}",
            ]
        )
    all_rows = []
    for i in val:
        cr, nr = cv_rows[i["id"]], nn_rows[i["id"]]
        all_rows.append(
            [
                i["id"],
                i["subset"],
                i["label"],
                cr["prediction"],
                cr["result"],
                nr["prediction"],
                nr["result"],
            ]
        )
    trial_rows = []
    for s in summary:
        dm = dev[s["model"]]["dev"]
        tm = dev[s["model"]]["train"]
        trial_rows.append(
            [
                s["model"],
                "да" if s["selected_on_dev"] else "нет",
                f"{tm['tp']}/6; FP {tm['fp']}/9",
                f"{dm['tp']}/3; FP {dm['fp']}/10",
                f"{s['counter_cash_detected']}/6; FP {s['false_cash_empty_counter']}/28",
                f"{s['threshold']:.6f}",
            ]
        )
    data_table = table(
        ["Выборка", "EMPTY", "CASH_PRESENT", "UNCERTAIN", "Примечание"],
        [
            ["X-1 + X-4: обучение", 9, 6, "5 исключены", "Ещё 13 почти одинаковых EMPTY исключены"],
            [
                "X-5: настройка/dev",
                10,
                3,
                "3 исключены",
                "Другой исходный ролик; пороги выбирались здесь",
            ],
            [
                "X-3 + X-2: фиксированная проверка",
                30,
                6,
                "6 без бинарной оценки",
                "Все 42 кадра обработаны",
            ],
        ],
    )
    metrics_table = table(
        [
            "Метод",
            "Деньги без руки",
            "Ложные cash на пустом",
            "Precision*",
            "Recall*",
            "Balanced accuracy*",
        ],
        main_rows,
    )
    six_table = table(["Кадр", "Метка", "OpenCV", "Score CV", "MobileNet", "Score NN"], six_rows)
    trials_table = table(
        ["Вариант", "Выбран на dev", "Train", "Dev", "Фиксированные 6 + 28", "Порог"], trial_rows
    )
    frame_table = table(
        ["Кадр", "Подгруппа", "Метка", "OpenCV", "Ошибка", "MobileNet", "Ошибка"], all_rows
    )
    html = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Counter Occupancy —
эксперимент</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;
max-width:1240px;
margin:40px auto;
padding:0 24px;
color:#17212b}}
h1,h2{{line-height:1.2}}h2{{margin-top:36px}}table{{border-collapse:collapse;
width:100%;
font-size:13px;
margin:18px 0}}
td,th{{padding:9px;
border:1px solid #d3dce3;
text-align:left}}th{{background:#edf2f6}}
img{{max-width:100%;
height:auto}}.notice{{background:#fff3da;
border-left:5px solid #b67700;
padding:18px}}
code{{background:#edf2f6;
padding:2px 5px}}pre{{background:#edf2f6;
padding:16px;
overflow:auto}}
details{{margin-top:24px}}.scroll{{overflow:auto}}small{{color:#52616c}}</style>
<h1>Counter Occupancy Detector</h1><p>Независимый эксперимент · 18 октября 2026 · ветка
<code>feat/counter-occupancy-experiment</code></p>
<p class="notice"><b>Решение: не подключать к Event Detector.</b> Выбранный на dev вариант OpenCV
обнаружил 2 из 6 кадров с деньгами без руки; MobileNet — 0 из 6. У обоих 0 ложных срабатываний
на 28 пустых кадрах. Главная цель пока не достигнута.</p>
<h2>Данные и границы эксперимента</h2>
<p>Использован существующий полигон <code>config/event_camera.json → zones.counter</code>:
<code>[[.07,.48],[.20,.47],[.25,.56],[.07,.59]]</code>. На исходных 464×848 кадрах ограничивающий
прямоугольник — 85×102 пикселя. Пиксели вне полигона занулены. Исходная зона не изменена.
В X-1, X-4, X-5 аппарат попадает в эту зону. Другие девять исходных обучающих роликов исключены
из-за несовпадения ракурса с зоной; список сохранён в annotations.json.</p>
{data_table}
<p>Извлечено 49 кропов. После исключения 8 неоднозначных кадров и 13 повторов EMPTY осталось
15 примеров обучения и 13 dev. Разнообразие ограничено: в обучении всего 6 положительных кадров
из двух циклов работы счётчика; верхний лоток на разных стадиях и нижний приёмник без руки.
Последовательные одинаковые EMPTY отбирались по минимальному отличию внутри полигона.
Разметка выполнена визуально ассистентом и является предварительной, без независимой проверки
человеком.</p>
<p><b>EMPTY:</b> видимые лотки пусты. <b>CASH_PRESENT:</b> видимая часть банкнот физически в верхнем
или нижнем лотке. <b>UNCERTAIN:</b> рука/загрузка/выгрузка не позволяют надёжно понять положение
денег.
Деньги или ID в руках рядом с аппаратом сами по себе не являются CASH_PRESENT.
Неоднозначные кадры не использованы для обучения или выбора порога.</p>
<h2>Методы</h2>
<ol><li>Медианный пустой эталон только из 9 обучающих EMPTY. Сравнение средней абсолютной
разности.</li>
<li>Разность в Lab с компенсацией общего сдвига яркости/цвета; 98-й перцентиль.</li>
<li>OpenCV-признаки по сетке 4×4: разность с эталоном, насыщенность, разброс яркости, текстура
по лапласиану. Линейный классификатор с регуляризацией. На dev выбрана регуляризация 10,
порог <code>{cv_selected["threshold"]:.6f}</code>.</li>
<li>MobileNetV3-Small с официальными ImageNet-весами, неизменяемый backbone и обученная бинарная
линейная голова. Кроп целиком дополнен до квадрата и увеличен до 224×224, без центрального
обрезания.
Признаки вычислены на NVIDIA RTX 3050 Laptop GPU. На dev выбрана регуляризация 10,
порог <code>{nn_selected["threshold"]:.6f}</code>. Это классификатор области, не новый
YOLO.</li></ol>
<p>Проверены только регуляризации 1/10/100. Выбор: balanced accuracy на X-5, затем меньше FP,
качество на train и более сильная регуляризация. CNN был запущен после неуспешной итоговой оценки
OpenCV, как разрешённый запасной подход. X-3/X-2 не участвовали в fit, выборе кропов обучения,
эталоне, нормализации или настройке порога. Параметры каждого метода сохранены до его итоговой
оценки.</p>
<h2>Фиксированная проверка</h2>{metrics_table}
<p><small>* Метрики по 36 однозначным кадрам: 6 CASH_PRESENT и 30 EMPTY. Из 8 кадров вне
обязательных
подгрупп 2 являются EMPTY, 6 — UNCERTAIN. Все 42 кадра имеют предсказание; для 6 UNCERTAIN бинарная
правильность не оценивается. «—» означает отсутствие положительных предсказаний, precision не
определён.
Это бинарная классификация заполненности; её precision нельзя напрямую сравнивать с bbox precision
YOLO.</small></p>
<p>Для контекста: в предыдущих трёх экспериментах YOLO деньги на этих шести кадрах обнаруживались
0/6.
OpenCV улучшил этот узкий показатель до 2/6, но этого недостаточно для использования.</p>
<h2>Ошибки на шести обязательных кадрах</h2>{six_table}
<p>В OpenCV пропущены v03_01850, v02_01550, v02_01600, v02_01650.
MobileNet пропустил все шесть. На обязательных 28 EMPTY ошибок нет.
Сырые scores не являются калиброванными вероятностями.</p>
<img src="validation-examples.jpg" alt="Шесть положительных кадров и два пустых: правильные и
ошибочные предсказания">
<h2>Что ограничивает результат</h2>
<p>На X-5 оба выбранных классификатора получили 3/3 CASH_PRESENT и 10/10 EMPTY, но это не
перенеслось
на X-3/X-2. В обучающих кропах банкноты преимущественно коричневые, в фиксированной проверке —
голубые;
меняются также освещение и отражения. Вероятная причина — узкое разнообразие данных и зависимость
признаков от внешнего вида. Это гипотеза по наблюдениям, не доказанная причинная диагностика.</p>
<p>Большую часть 85×102 кропа занимает корпус; видимая часть денег в верхнем лотке — около 18×16
пикселей. Увеличение до 224×224 не добавляет деталей. Пустые кадры не покрывают широкий набор
помех: например, ID лежащего непосредственно в лотке здесь нет. Поэтому 0/28 не доказывает
отсутствие ложных срабатываний в других сценах. Кадры внутри каждого ролика коррелированы.</p>
<img src="training-examples.jpg" alt="Реальные обучающие кропы: верхний и нижний лоток, пустой
аппарат">
<h2>Подключение</h2>
<p><b>Интеграция не выполнена и сейчас не рекомендуется.</b> Увеличивать число кадров копированием
или вымышленной разметкой не стали. Для следующей проверки нужны отдельные обучающие эпизоды
с разными банкнотами в верхнем/нижнем лотке без руки и сложными отрицательными примерами
при этом ракурсе, плюс новые ролики для независимой проверки.</p>
<p>Если дальнейший кандидат пройдёт проверку, минимальная роль — дополнительное наблюдение
<code>CounterOccupancyObservation(timestamp, zone_id, state, score, visibility,
model_version)</code>.
Оно должно только подтверждать состояние области. Закрытие рукой должно приводить к неизвестному
состоянию, а не EMPTY; нынешний бинарный классификатор сам надёжно окклюзию не распознаёт.
События MONEY_TO_COUNTER/MONEY_FROM_COUNTER требуют согласованных наблюдений рук/денег,
треков и переходов существующего Event Detector. Переключение occupancy само по себе событием не
является.</p>
<details><summary>Все варианты — без выбора лучшего по validation</summary><div
class="scroll">{trials_table}</div></details>
<details><summary>Покадровая таблица всех 42 кадров</summary><div
class="scroll">{frame_table}</div></details>
<h2>Артефакты и повторная проверка</h2>
<p>Разметка и отчёты: <code>datasets/counter_occupancy_v1/</code>. Кропы, пустой эталон,
JSON-параметры классификаторов, официальные .pth-веса и этот отчёт:
<code>data/counter_occupancy_v1/</code>. Веса YOLO, tracking, Event Detector, Rule Engine
и конфигурация сохранены: проверены SHA-256 59 исходных файлов; все 42 исходных
validation-изображения совпадают.</p>
<pre>./.venv/Scripts/python.exe experiments/counter_occupancy.py audit
./.venv/Scripts/python.exe experiments/counter_occupancy.py evaluate-opencv
./.venv/Scripts/python.exe -m experiments.counter_occupancy_cnn evaluate
./.venv/Scripts/python.exe -m experiments.counter_occupancy_report</pre>
<p>Эти команды оценивают сохранённые кандидаты. Полный pipeline не запускался. Основные веса
не заменялись, события не создавались.</p></html>"""
    (DATA / "report.html").write_text(html, encoding="utf-8")
    artifacts = [
        DATA / "opencv-models.json",
        DATA / "cnn-models.json",
        DATA / "empty-reference.png",
        META / "annotations.json",
        META / "validation.json",
    ]
    write_json(
        META / "experiment-artifacts-sha256.json",
        {str(p.relative_to(ROOT)): sha(p) for p in artifacts},
    )
    print(str(Path(DATA / "report.html")))


if __name__ == "__main__":
    main()
