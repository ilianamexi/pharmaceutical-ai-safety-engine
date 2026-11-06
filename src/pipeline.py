"""Case-intake triage demo: score adverse-event reports with the trained models.

For each report: map drugs to DrugBank, score drug pairs with the BioSNAP GCN, predict the
probability that the report is serious, and rank reports for review.
This is a research/portfolio demo and is NOT validated for regulatory or clinical use.

Usage:
    # score reports you already downloaded (note: these were used for training -> in-sample)
    python -m src.pipeline --input data/raw/openfda_events.jsonl
    # fetch NEW reports from a later window that the model never saw, then score them
    python -m src.pipeline --fetch 20250101 20250131 --n 200
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from src.data.drug_mapping import load_mapper
from src.data.openfda import api_key_from_env, fetch_reports, load_reports
from src.models.features import ReportFeaturizer
from src.models.focal_risk_engine import RiskMLP
from src.models.train_risk_model import build_text_embedder, load_graph_features
from src.utils.common import load_config, resolve


class TriagePipeline:
    def __init__(self, config_path=None):
        self.cfg = load_config(config_path)
        ckpt_path = resolve(self.cfg["paths"]["models_dir"]) / "risk_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError("Train first: python -m src.gnn.train_ddi_gnn && python -m src.models.train_risk_model")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        f = ck["featurizer"]
        cfg = dict(ck["config"])
        cfg["risk_model"] = {**cfg["risk_model"], "features": f["features"]}

        mapper = load_mapper(resolve(cfg["paths"]["drugbank_vocabulary"])) if "gnn" in f["features"] else None
        self.graph = load_graph_features(cfg) if "gnn" in f["features"] else None
        text_fn = build_text_embedder(cfg) if "text" in f["features"] else None
        self.featurizer = ReportFeaturizer(cfg, mapper, self.graph, text_fn)
        self.featurizer.drug_vocab, self.featurizer.reaction_vocab = f["drug_vocab"], f["reaction_vocab"]
        self.featurizer.age_mean, self.featurizer.age_std = f["age_mean"], f["age_std"]

        self.model_name, self.threshold, self.target = ck["model_name"], ck["threshold"], cfg["risk_model"]["target"]
        if self.model_name == "logistic_regression":
            self.model = ck["model"]
        else:
            self.model = RiskMLP(ck["in_dim"], cfg["risk_model"]["hidden_dim"], cfg["risk_model"]["dropout"])
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_name == "logistic_regression":
            return self.model.predict_proba(X)[:, 1]
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(X))).numpy()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        X, info = self.featurizer.transform(df)
        p = self.predict_proba(X)
        out = pd.DataFrame({
            "safetyreportid": df["safetyreportid"].values,
            "receivedate": df["receivedate"].values,
            "suspect_drugs": ["; ".join(d.get("active_substance") or d["name"] for d in drugs if d["role"] == "suspect")
                              for drugs in df["drugs"]],
            "n_drugs": df["drugs"].map(len).values,
            "reactions": df["reactions"].map(lambda r: "; ".join(r[:5])).values,
            "n_drugs_in_ddi_graph": [sum(i in self.graph.node_index for i in ids) if self.graph else 0
                                     for ids in info["mapped_drugbank_ids"]],
            "top_ddi_pair": [" + ".join(pr) if pr else "" for pr in info["top_ddi_pair"]],
            "top_ddi_prob": info["top_ddi_prob"].values,
            "top_pair_known_in_biosnap": [bool(self.graph.is_known(*pr)) if pr else None for pr in info["top_ddi_pair"]],
            f"p_{self.target}": np.round(p, 4),
            "priority": np.where(p >= self.threshold, "HIGH - review first", "ROUTINE"),
        })
        if self.target in df.columns:
            out[f"actual_{self.target}"] = df[self.target].astype(bool).values
        return out.sort_values(f"p_{self.target}", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="JSONL produced by src.data.download")
    parser.add_argument("--fetch", nargs=2, metavar=("DATE_FROM", "DATE_TO"), help="fetch new reports, YYYYMMDD")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output", default="results/triage_output.csv")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    pipe = TriagePipeline(args.config)
    if args.fetch:
        path = resolve(f"data/raw/openfda_{args.fetch[0]}_{args.fetch[1]}.jsonl")
        fetch_reports(path, args.fetch[0], args.fetch[1], n_reports=args.n, page_size=min(args.n, 1000),
                      api_key=api_key_from_env(pipe.cfg["openfda"]["api_key_env"]))
    else:
        path = resolve(args.input or pipe.cfg["paths"]["openfda_raw"])
        print("[pipeline] NOTE: scoring reports that may overlap the training data (in-sample).")
    df = load_reports(path)
    if args.n:
        df = df.head(args.n)

    out = pipe.run(df)
    out_path = resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 40)
    print(out.head(15).to_string(index=False))
    print(f"\n[pipeline] model={pipe.model_name}, threshold={pipe.threshold:.3f} (chosen on validation)")
    print(f"[pipeline] {int((out['priority'] != 'ROUTINE').sum())}/{len(out)} reports flagged HIGH -> {out_path}")


if __name__ == "__main__":
    main()
