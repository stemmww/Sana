# Cash / ID detector experiment

## Результат: кандидат отклонён

Проведены аудит, исправление разметки, обучение двух классов и сравнение на неизменных
42 validation-кадрах. Production weights не заменены. Rule Engine, Event Detector,
tracking, зоны и temporal thresholds не изменены.

| Метрика | OLD | NEW best | NEW last |
| --- | ---: | ---: | ---: |
| Cash precision | 52.9% | 58.3% | 55.0% |
| Cash recall | 31.0% | 48.3% | 37.9% |
| Cash TP / FP / FN | 9 / 8 / 20 | 14 / 10 / 15 | 11 / 9 / 18 |
| ID precision | — | 64.0% | 55.2% |
| ID recall | 0% | 72.7% | 72.7% |
| ID TP / FP / FN | 0 / 0 / 22 | 16 / 9 / 6 | 16 / 13 / 6 |
| ID → cash, покрыто ≥50% площади ID | 3 | 2 | 3 |
| ID → cash, строгий IoU ≥0.5 | 2 | 0 | 3 |
| Cash FP в ROI пустого счётчика, 28 кадров | 0 | 1 | 2 |

Старая модель не поддерживает ID: precision не определён, recall равен нулю.
У `best` один FP в зоне пустого аппарата — это ID около его верхнего края,
ошибочно названный cash (`v03_00600`), а не доказательство распознавания корпуса как денег.
В `v03_01150` ID при укладывании на стол также назван cash. У `last` два cash-box
на одном проблемном кадре около счётчика.

| Диагностическая категория | OLD precision / recall | NEW best precision / recall |
| --- | ---: | ---: |
| cash_in_hand, 10 объектов | 57.1% / 40.0% | 30.0% / 30.0% |
| cash_transfer, 5 объектов | 80.0% / 80.0% | 80.0% / 80.0% |
| cash_at_counter, 14 объектов | 50.0% / 7.1% | 87.5% / 50.0% |

**Все 7 успешных counter detections нового best относятся к поднесению/загрузке денег.**
На шести кадрах после отхода руки кассира новая модель пропускает пачку:
`v03_01850`, `v03_01900`, `v03_01950`, `v02_01550`, `v02_01600`, `v02_01650`.
Это данные аудита, не условия inference. Положительная общая динамика категории не решает
задачу стабильного обнаружения денег внутри счётчика.

**Решение:** best и last не проходят критерии принятия. Лучшего допустимого нового
детектора не выбрано, поэтому новый прогон X3 → X1 → X2 не выполнялся согласно условию
пользователя. Предыдущие pipeline-результаты сохранены; улучшение событий не заявляется.

Следующая необходимая работа в dataset: дополнительные независимые от validation видео
с видимой пачкой в счётчике без руки, разными видами банкнот, ID около аппарата и ID при
укладывании на стол. В текущем обучении counter-примеры преимущественно с коричневыми
банкнотами; в validation пачки голубые. Дополнительно нужны деньги в руках под другими
углами: эта категория ухудшилась. Validation-кадры переносить в train нельзя.

Артефакты:

- `data/cash_id/evaluation/{old,new,last}/metrics.json`: все предсказания и метрики.
- `data/cash_id/evaluation/{new,last}/comparison.json`: критерии и отказ в принятии.
- `data/cash_id/comparison.jpg`: сравнение одинаковых кадров.
- `data/cash_id/model-card.json`: параметры и контрольные суммы.
- `data/cash/runs/cash_id_v1_run2/weights/best.pt`: экспериментальные веса, не production.

Проверки: 72 теста экспорта/метрик/существующих событий и правил; Ruff без ошибок.
Это небольшая предварительно размеченная выборка текущей камеры, а не независимый тест
готовности к эксплуатации.

## Scope and audit

This experiment changes dataset preparation, standalone object detection and evaluation only.
Production weights, `app/events`, `app/rules`, the video pipeline, person association,
camera geometry and temporal thresholds are preserved. Their before/after hashes are recorded
in `data/cash_id/frozen-sha256.json`.

The original manifest contains 58 frames, of which 49 were reviewed provisionally:
33 train, 9 validation and 7 test. All 58 were visually screened. The original cash-only
annotations cannot serve as complete two-class labels. There were no exact duplicate boxes
or out-of-bounds boxes in the original manifest; that does not establish semantic correctness.

| Finding | Evidence | Disposition |
| --- | --- | --- |
| Missing ID annotations | `x01_008`, `x04_015`; ID on table after the handover | Re-annotate the sequence with `id_card` |
| Cash labelled as background | `x04_030`, also `x05_030`, visible cash in upper counter tray | Replace the negative with a cash box; add table ID |
| Excluded useful positive | `x01_018`, cash visible in counter | Include reviewed neighboring samples |
| Ambiguous cash identity/boundary | `x05_015`, small box by hand/wallet, visible ID elsewhere | Exclude pending separate sequence review |
| Stale counter label in previous experiment | X4 approximately 31 seconds: upper tray already empty, old experimental box remains there | Exclude uncertain output-tray views; never copy a box across time without review |
| Sparse temporal/state coverage | Handovers and counter use sampled infrequently | Review denser samples covering approach, inlet, visible cash, withdrawal and subsequent holding |
| No diagnostic coverage metadata | All objects are simply cash | Add per-object categories and reviewed negative-frame tags |

The item-by-item disposition is in `datasets/cash_id_v1/legacy-audit.json`.
Original annotations and the rejected counter-only experiment are retained unchanged.
15 legacy frames were retained after visual review; X1/X4 were resampled and re-annotated;
remaining legacy frames are excluded pending their own two-class sequence review.

### Annotation policy

- Exactly two YOLO classes, in this order: `0: cash`, `1: id_card`.
- Boxes cover the visible object, excluding the hand, wallet or machine casing where feasible.
- One box per contiguous stack/fan. No assertion about denomination, amount or ownership.
- `cash_in_hand`, `cash_transfer`, `cash_at_counter`, `cash_on_table` are diagnostic categories.
- ID categories describe customer hand, handover, cashier hand and table visibility.
- Unresolved identity, motion blur or severe occlusion excludes a frame, not a negative label.
- `empty_counter` requires visual review and an explicit ROI for evaluation. That ROI is
  never an input to the object detector or an inference rule.
- The annotations are assistant-reviewed provisional labels, not independent human gold labels.

## Frozen data

`datasets/cash_id_v1/annotations.json` is the authoritative corrected manifest.
Frame pixels and exported YOLO data are local, ignored artifacts under `data/cash_id`.

| Split | Images | Source videos | Cash boxes | ID boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 75 | 9 | 60 | 53 |
| Validation | 42 | 2 | 29 | 22 |

14 newly sampled ambiguous frames are excluded. Train includes X1/X4 and reviewed legacy
frames from other training sources. Validation uses X3/X2, absent from both the original
model's training sources and this candidate's training sources. X1 is a development replay,
not an independent detector validation video. The old validation/test sources are not moved
into this training set.

Validation has 10 cash-in-hand, 5 cash-transfer and 14 cash-at-counter targets, 22 ID targets,
28 empty-counter frames, and 13 no-cash frames with hands/table visible. Seven frames contain
ID without cash. These categories overlap and are not separate YOLO classes.

Frames from one video stay in one split. Source hashes, image hashes, image dimensions,
duplicate pixels and bbox bounds are checked. The validation seal includes pixel hashes,
labels, categories, source identity and sampling metadata. Any change invalidates the seal.
Nearby frames remain correlated: these results cover the current camera/scenarios and do not
establish generalization to a different room, camera or an independent test set.

## Reproduce locally

Run from the repository root. Export destinations and run names must be fresh.

```powershell
.\.venv\Scripts\python.exe -m app.cv.object_dataset datasets/cash_id_v1/annotations.json --output data/cash_id/dataset_v1
.\.venv\Scripts\python.exe -m app.cv.train_cash --data data/cash_id/dataset_v1/objects.yaml --base data/cash/models/cash.pt --epochs 30 --imgsz 768 --batch 4 --name cash_id_v1_run2
.\.venv\Scripts\python.exe -m app.cv.evaluate_objects --annotations datasets/cash_id_v1/annotations.json --model data/cash/models/cash.pt --output data/cash_id/evaluation/old
.\.venv\Scripts\python.exe -m app.cv.evaluate_objects --annotations datasets/cash_id_v1/annotations.json --model data/cash/runs/cash_id_v1_run2/weights/best.pt --output data/cash_id/evaluation/new --compare-to data/cash_id/evaluation/old/metrics.json
```

The first training attempt was blocked by Windows sandbox multiprocessing permissions;
`cash_id_v1_run2` is the actual local training run. Training uses the existing deterministic
training configuration: frozen first 10 layers, AdamW, seed 42, CPU, 768 pixels, 30 epochs.
The validation images do not participate in training batches/augmentation or gradient updates.
Ultralytics uses validation to select `best.pt`; this is model selection, not an untouched test.
No default weights are copied or overwritten.

`ObjectDetector` returns both labels from two-class weights. Existing `CashDetector` can
load the same weights and select only the `cash` class for unchanged downstream replay.
Detecting a card does not establish `ID_CHECK`, its owner or completion of an ID procedure.
The current real-video event adapter's ID-sequence coverage remains disabled.

## Comparison protocol

Both models use full original frames, image size 768, confidence 0.25 and matching IoU 0.5.
Matching is class-aware, confidence-ordered and one-to-one. Duplicate detections are FPs.
Missing classes have zero recall; precision is undefined when there are no predictions.

- Empty-counter FP: unmatched cash box whose center falls within the reviewed empty ROI.
- ID-to-cash FP: unmatched cash box covering >=50% of an ID annotation's area.
- Strict ID-to-cash localization diagnostic: unmatched cash box with IoU >=0.5 to ID.
- Category recall: matched objects / annotated objects of that category.
- Category precision: category TP / (category TP + unmatched cash predictions on its frames).
  It is a conditional diagnostic; it is not precision of a separate class.
- No-cash hands/table FP counts cover the entire tagged frame, not a segmented hand/table ROI.

Acceptance gates, fixed before candidate evaluation:

1. Counter recall improves.
2. Cash FP count increases by at most one; cash precision falls by no more than 5 percentage points.
3. Overall cash recall does not fall.
4. ID-to-cash confusions halve or remain zero.
5. Zero empty-counter FPs on this validation set.
6. At least one correct ID detection.

Passing these local gates permits an explicit pipeline trial, not a claim of production readiness.
The report records raw TP/FP/FN, denominators, hashes, predictions and annotated previews.

### Metric audit after visual inspection

The first implementation counted ID confusions only at IoU >=0.5. It reported 2 → 0
confusions, but inspection revealed two loose cash boxes over small IDs in the candidate.
Metrics version 2 adds ID-area coverage and applies it identically to all three cached
prediction sets: 3 → 2 → 3 confusions. Detection TP/FP/FN matching remains IoU >=0.5;
labels, images and confidence are unchanged. Original IoU-only reports are preserved as
`metrics-iou-only.json`. The candidate failed acceptance under both metric versions.
The same acceptance limits are used; the confusion measurement was corrected, not relaxed.

## Video interpretation

X3 ends with cash in the counter, as confirmed by the user. Its observed target is
`MONEY_TRANSFER_TO_CASHIER -> MONEY_TO_COUNTER`; withdrawal and subsequent holding cannot
be inferred from that clip. X1 contains withdrawal and return to the customer; X2 has no
visible ID procedure. Detector improvements alone may still leave association/temporal
limitations. Hand association and business-rule changes are outside this experiment.
