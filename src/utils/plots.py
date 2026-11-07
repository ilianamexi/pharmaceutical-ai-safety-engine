"""Static figures for the README (PNG, light surface).

Usage:  python -m src.utils.plots      # draws every figure whose input file exists
Inputs are the prediction / metrics files written by the training and evaluation scripts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils.common import load_config, resolve

# Categorical slots 1-3 of the reference palette (validated for colour-vision deficiency);
# lines also differ in dash style, so identity never relies on colour alone.
SERIES = [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", ":")]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2,
    })
    return plt


def _roc_pr_axes(ax_roc, ax_pr, y, curves: dict[str, np.ndarray], title_suffix=""):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    for (name, s), (color, dash) in zip(curves.items(), SERIES):
        fpr, tpr, _ = roc_curve(y, s)
        ax_roc.plot(fpr, tpr, color=color, linestyle=dash, label=f"{name}  (AUROC {roc_auc_score(y, s):.3f})")
        p, r, _ = precision_recall_curve(y, s)
        ax_pr.plot(r, p, color=color, linestyle=dash, label=f"{name}  (AUPRC {average_precision_score(y, s):.3f})")
    ax_roc.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=(0, (2, 3)))
    ax_pr.axhline(np.mean(y), color=AXIS, linewidth=1, linestyle=(0, (2, 3)))
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01),
               title=f"ROC{title_suffix}")
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01), title=f"Precision–recall{title_suffix}")
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="lower right" if ax is ax_roc else "lower left", fontsize=9)
        ax.set_aspect("equal")


def plot_ddi(pred_csv, out_png):
    """ROC curves of the GCN vs Adamic-Adar under random and degree-matched negatives."""
    from sklearn.metrics import roc_auc_score, roc_curve

    plt = _style()
    df = pd.read_csv(pred_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    titles = {"random_negatives": "Random negatives", "degree_matched_negatives": "Degree-matched negatives (harder)"}
    for ax, (setting, title) in zip(axes, titles.items()):
        d = df[df["setting"] == setting]
        for (col, label), (color, dash) in zip([("gcn", "GCN"), ("adamic_adar", "Adamic–Adar")], SERIES):
            fpr, tpr, _ = roc_curve(d["y"], d[col])
            ax.plot(fpr, tpr, color=color, linestyle=dash, label=f"{label}  (AUROC {roc_auc_score(d['y'], d[col]):.3f})")
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=(0, (2, 3)))
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01), title=title)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Drug–drug interaction prediction, BioSNAP test edges (seed 42)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_seriousness(pred_csv, out_png):
    plt = _style()
    df = pd.read_csv(pred_csv)
    names = {"logistic_regression": "Logistic regression", "mlp_bce": "MLP + BCE", "mlp_focal": "MLP + focal"}
    curves = {label: df[col].values for col, label in names.items() if col in df}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6))
    _roc_pr_axes(a, b, df["y"].values, curves)
    fig.suptitle(f"Adverse-event seriousness, openFDA FAERS test set (n = {len(df):,})", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_temporal(metrics_json, out_png):
    """AUROC per month of unseen reports, with the in-period test AUROC as a reference line."""
    plt = _style()
    m = json.load(open(metrics_json, encoding="utf-8"))
    months = m["per_month"]
    x = np.arange(len(months))
    auc = np.array([r["auroc"] for r in months])
    lo = np.array([r["auroc_95ci"][0] for r in months])
    hi = np.array([r["auroc_95ci"][1] for r in months])
    fig, ax = plt.subplots(figsize=(8, 4))
    color = SERIES[0][0]
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, label="95% bootstrap CI")
    ax.plot(x, auc, color=color, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="AUROC on unseen month")
    ref = m.get("in_period_test_auroc")
    if ref:
        ax.axhline(ref, color=INK_2, linewidth=1, linestyle="--", label=f"In-period test AUROC ({ref:.3f})")
    ax.set_xticks(x, [r["month"] for r in months])
    ax.set(ylabel="AUROC", ylim=(max(0.5, float(lo.min()) - 0.05), 1.0),
           title=f"Temporal validation: trained on reports received {'–'.join(w[:4] for w in m['training_window'])}, "
                 f"scored on later months")
    for xi, yi, r in zip(x, auc, months):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8, color=INK_2)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def make_all(cfg=None) -> list[str]:
    cfg = cfg or load_config()
    res = resolve(cfg["paths"]["results_dir"])
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    made = []
    jobs = [
        (res / "gnn_test_predictions.csv", figs / "ddi_roc.png", plot_ddi),
        (res / "risk_model_test_predictions.csv", figs / "seriousness_roc_pr.png", plot_seriousness),
        (res / "temporal_check.json", figs / "temporal_auroc.png", plot_temporal),
    ]
    for src, dst, fn in jobs:
        if src.exists():
            try:
                fn(src, dst)
                made.append(str(dst))
            except (KeyError, ValueError) as e:  # e.g. an older metrics file without per-month data
                print(f"[plots] skipped {dst.name}: {e}")
    return made


if __name__ == "__main__":
    for p in make_all():
        print(f"[plots] wrote {p}")
