"""Shared helpers: config loading, seeding, metrics, JSON output."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root (works from any cwd)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def binary_metrics(y_true, y_score, threshold: float | None = None) -> dict:
    """AUROC / AUPRC, plus precision/recall/F1 when a decision threshold is given."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    out = {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }
    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        out.update(
            threshold=float(threshold),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
        )
    return out


def best_f1_threshold(y_true, y_score) -> float:
    """Threshold maximising F1 — must be chosen on the validation split, never on test."""
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(y_score))
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    # precision_recall_curve returns one more (p, r) point than thresholds
    return float(t[int(np.nanargmax(f1[:-1]))])


def bootstrap_ci(y_true, y_score, metric_fn, n_iter: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for a ranking metric."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    stats = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # resample contains one class only
        stats.append(metric_fn(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def save_json(obj, path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
