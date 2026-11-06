"""Runs the full risk-model + triage path on SYNTHETIC data with a tiny GNN checkpoint."""
import numpy as np
import pandas as pd
import torch

from src.data.drug_mapping import DrugBankMapper
from src.data.openfda import load_reports
from src.models.features import GraphFeatures, ReportFeaturizer
from src.models.train_risk_model import fit_and_score_all
from src.utils.common import load_config
from tests.synthetic import VOCAB_ROWS, write_reports, write_vocab


def test_risk_model_runs_end_to_end(tmp_path):
    cfg = load_config()
    cfg["risk_model"].update(max_epochs=3, patience=2, focal_alpha_grid=[0.5], focal_gamma_grid=[2.0],
                             bootstrap_iterations=50, top_k_drugs=20, top_k_reactions=20)
    write_reports(tmp_path / "r.jsonl", n=600)
    write_vocab(tmp_path / "v.csv")
    df = load_reports(tmp_path / "r.jsonl")

    node_ids = [r[0] for r in VOCAB_ROWS]
    graph = GraphFeatures({"node_ids": node_ids, "embeddings": torch.randn(len(node_ids), 8)},
                          {("DB00945", "DB00682")})
    feat = ReportFeaturizer(cfg, DrugBankMapper(tmp_path / "v.csv"), graph).fit(df.iloc[:400])
    X, info = feat.transform(df)
    assert X.shape[0] == len(df) and np.isfinite(X).all()
    assert "gnn" in feat.features
    # a report containing aspirin + warfarin must expose the known BioSNAP pair
    has_pair = df["drugs"].map(lambda ds: {"ASPIRIN", "WARFARIN SODIUM"} <= {d["name"] for d in ds})
    if has_pair.any():
        assert info.loc[has_pair, "n_known_ddi_pairs"].min() >= 1

    y = df["serious"].astype(int).values
    res, fitted = fit_and_score_all(X[:400], y[:400], X[400:500], y[400:500], X[500:], y[500:], cfg, 0, verbose=False)
    for name in ("logistic_regression", "mlp_bce", "mlp_focal"):
        assert 0.0 <= res[name]["test"]["auroc"] <= 1.0
        assert "threshold" in res[name]["val"]


def test_leakage_terms_are_removed(tmp_path):
    cfg = load_config()
    cfg["risk_model"]["target"] = "death"
    cfg["risk_model"]["features"] = ["reactions"]
    write_reports(tmp_path / "r.jsonl", n=200)
    feat = ReportFeaturizer(cfg).fit(load_reports(tmp_path / "r.jsonl"))
    assert "death" not in feat.reaction_vocab
