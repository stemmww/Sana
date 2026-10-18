"""Optional frozen ImageNet MobileNetV3-Small + fitted binary linear head.

Only the classification head is fitted. No YOLO code/weights or runtime events.
"""

import argparse
import json
import time

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from experiments.counter_occupancy import (
    DATA,
    META,
    ROOT,
    binary_metrics,
    checked_manifest,
    evaluate_scores,
    imread,
    model_scores,
    read_json,
    select_threshold,
    selected,
    sha,
    write_json,
)

WEIGHTS_PATH = DATA / "pretrained/mobilenet_v3_small-047dcff4.pth"


def download():
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = MobileNet_V3_Small_Weights.DEFAULT.url
    if not WEIGHTS_PATH.exists():
        torch.hub.download_url_to_file(
            url, str(WEIGHTS_PATH), hash_prefix="047dcff4", progress=False
        )
    assert sha(WEIGHTS_PATH).startswith("047dcff4")
    print(json.dumps({"url": url, "file": str(WEIGHTS_PATH), "sha256": sha(WEIGHTS_PATH)}))


def backbone():
    assert WEIGHTS_PATH.is_file(), "Run download first."
    assert sha(WEIGHTS_PATH).startswith("047dcff4")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requests the available RTX GPU; CUDA is unavailable.")
    net = mobilenet_v3_small(weights=None)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    net.classifier = torch.nn.Identity()
    net.requires_grad_(False).eval().cuda()
    return net


def preprocess(path):
    crop = cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)
    h, w = crop.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), np.uint8)
    padded[(size - h) // 2 : (size - h) // 2 + h, (size - w) // 2 : (size - w) // 2 + w] = crop
    x = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1))


def embeddings(net, items):
    output = []
    with torch.inference_mode():
        for start in range(0, len(items), 16):
            batch = torch.stack(
                [preprocess(ROOT / i["crop"]) for i in items[start : start + 16]]
            ).cuda()
            output.append(net(batch).cpu().numpy())
    return np.concatenate(output)


def fit():
    manifest = checked_manifest()
    train, dev = selected(manifest["items"], "train"), selected(manifest["items"], "dev")
    net = backbone()
    start = time.perf_counter()
    feats = embeddings(net, train + dev)
    n = len(train)
    mean, scale = feats[:n].mean(axis=0), feats[:n].std(axis=0).clip(0.1)
    x = (feats - mean) / scale
    y = np.array([1 if i["label"] == "CASH_PRESENT" else -1 for i in train], np.float32)
    weights = np.array([n / (2 * np.sum(y == label)) for label in y])
    models = []
    for alpha in [1.0, 10.0, 100.0]:
        xx = np.column_stack([x, np.ones(len(x))])
        penalty = np.eye(xx.shape[1]) * alpha
        penalty[-1, -1] = 0
        coef = np.linalg.solve((xx[:n].T * weights) @ xx[:n] + penalty, (xx[:n].T * weights) @ y)
        scores = xx @ coef
        threshold, metrics = select_threshold(dev, scores[n:])
        models.append(
            {
                "name": f"mobilenet_v3_small_frozen_ridge_{alpha:g}",
                "type": "ridge",
                "alpha": alpha,
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "coef": coef.tolist(),
                "threshold": threshold,
                "train": binary_metrics(train, scores[:n], threshold),
                "dev": metrics,
            }
        )
    best = max(
        models,
        key=lambda m: (
            m["dev"]["balanced_accuracy"],
            -m["dev"]["fp"],
            m["train"]["balanced_accuracy"],
            m["alpha"],
        ),
    )
    artifact = {
        "architecture": (
            "MobileNetV3-Small frozen ImageNet backbone, fitted class-balanced ridge head"
        ),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "elapsed_sec": time.perf_counter() - start,
        "backbone_sha256": sha(WEIGHTS_PATH),
        "backbone_url": MobileNet_V3_Small_Weights.DEFAULT.url,
        "annotation_sha256": sha(META / "annotations.json"),
        "train_ids": [i["id"] for i in train],
        "dev_ids": [i["id"] for i in dev],
        "preprocess": (
            "Polygon-masked RGB, square black padding, bilinear 224x224, "
            "ImageNet normalization. No center crop."
        ),
        "selection": (
            "Dev balanced accuracy, fewer dev false positives, train balanced accuracy, "
            "stronger regularization. No validation used."
        ),
        "fallback_reason": (
            "Frozen OpenCV candidate detected only 2/6 required validation positives; "
            "fallback was requested by the user for insufficient OpenCV quality."
        ),
        "models": models,
        "selected_model": best["name"],
    }
    write_json(DATA / "cnn-models.json", artifact)
    write_json(
        META / "cnn-selection-frozen.json",
        {
            "model_sha256": sha(DATA / "cnn-models.json"),
            "selected_model": best["name"],
            "train_annotation_sha256": sha(META / "annotations.json"),
            "opencv_validation_seen_before_fallback": True,
            "fixed_validation_used_for_fit_or_threshold": False,
        },
    )
    summary = {
        "device": artifact["device"],
        "elapsed_sec": artifact["elapsed_sec"],
        "selected_model": best["name"],
        "models": [
            {k: v for k, v in m.items() if k not in {"mean", "scale", "coef"}} for m in models
        ],
    }
    write_json(META / "cnn-dev-results.json", summary)
    print(json.dumps(summary, indent=2))


def evaluate():
    checked_manifest()
    artifact = read_json(DATA / "cnn-models.json")
    assert (
        sha(DATA / "cnn-models.json")
        == read_json(META / "cnn-selection-frozen.json")["model_sha256"]
    )
    assert artifact["annotation_sha256"] == sha(META / "annotations.json")
    assert artifact["backbone_sha256"] == sha(WEIGHTS_PATH)
    items = read_json(META / "validation.json")["items"]
    net = backbone()
    features = embeddings(net, items)
    evaluations = [evaluate_scores(items, m, model_scores(m, features)) for m in artifact["models"]]
    write_json(
        META / "cnn-validation-results.json",
        {
            "frozen_model_sha256": sha(DATA / "cnn-models.json"),
            "selected_model": artifact["selected_model"],
            "evaluations": evaluations,
        },
    )
    chosen = next(e for e in evaluations if e["model"] == artifact["selected_model"])
    print(
        json.dumps(
            {"selected_model": artifact["selected_model"], "metrics": chosen["metrics"]}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["download", "fit", "evaluate"])
    args = parser.parse_args()
    {"download": download, "fit": fit, "evaluate": evaluate}[args.command]()
