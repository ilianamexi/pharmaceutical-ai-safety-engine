"""Train and evaluate the GCN link predictor on the BioSNAP ChCh-Miner DDI network.

Protocol
- Edges are split 80/10/10 (train/val/test) with PyG RandomLinkSplit; val/test edges are
  hidden from message passing during training.
- Val/test negatives are an equal number of random drug pairs not in the graph (1:1),
  so the AUPRC of a random scorer is ~0.5.
- Fresh negatives are sampled every training epoch; early stopping on validation AUROC.
- Classic link-prediction heuristics computed on the same graph are reported as baselines.
  Preferential attachment (degree product) shows how much of the signal is just node degree.

Usage:  python -m src.gnn.train_ddi_gnn [--seeds 42 43 44]
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling, to_undirected

from src.gnn.ddi_graph_model import DDIGraphModel
from src.utils.common import bootstrap_ci, load_config, resolve, save_json, set_seed


def load_graph(edges_csv) -> tuple[Data, list[str]]:
    df = pd.read_csv(edges_csv)
    node_ids = sorted(pd.unique(df[["Drug_A_ID", "Drug_B_ID"]].values.ravel()).tolist())
    idx = {d: i for i, d in enumerate(node_ids)}
    src = torch.tensor(df["Drug_A_ID"].map(idx).values)
    dst = torch.tensor(df["Drug_B_ID"].map(idx).values)
    edge_index = to_undirected(torch.stack([src, dst]), num_nodes=len(node_ids))
    return Data(edge_index=edge_index, num_nodes=len(node_ids)), node_ids


def heuristic_scores(edge_index: torch.Tensor, num_nodes: int, pairs: torch.Tensor) -> dict[str, np.ndarray]:
    ei = edge_index.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(num_nodes, num_nodes))
    A.data[:] = 1.0  # binarise in case of duplicates
    deg = np.asarray(A.sum(axis=1)).ravel()
    u, v = pairs.numpy()
    cn = np.asarray(A[u].multiply(A[v]).sum(axis=1)).ravel()
    inv_log = sp.diags(1.0 / np.log(np.clip(deg, 2, None)))
    aa = np.asarray((A[u] @ inv_log).multiply(A[v]).sum(axis=1)).ravel()
    return {
        "preferential_attachment": deg[u] * deg[v],
        "common_neighbors": cn,
        "adamic_adar": aa,
    }


def degree_matched_negatives(pos: torch.Tensor, mp_edge_index: torch.Tensor, all_edge_index: torch.Tensor,
                             num_nodes: int, seed: int) -> torch.Tensor:
    """Harder negatives that remove the node-degree shortcut.

    For every positive pair (u, v) one endpoint is kept and the other is replaced by a drug w drawn
    with probability proportional to its degree in the message-passing graph, so negative endpoints
    are as "popular" as positive endpoints. Pairs that are known interactions (in the full graph)
    are rejected.
    """
    rng = np.random.default_rng(seed)
    known = set(map(tuple, all_edge_index.t().tolist()))
    deg = np.bincount(mp_edge_index[0].cpu().numpy(), minlength=num_nodes).astype(float)
    prob = deg / deg.sum()
    out = []
    for u, v in pos.t().tolist():
        keep = u if rng.random() < 0.5 else v
        while True:
            w = int(rng.choice(num_nodes, p=prob))
            if w != keep and (keep, w) not in known:
                out.append((keep, w))
                break
    return torch.tensor(out, dtype=torch.long).t()


@torch.no_grad()
def score_pairs(model, edge_index, pairs) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(edge_index, pairs)).cpu().numpy()


@torch.no_grad()
def evaluate(model, data) -> tuple[float, float, np.ndarray]:
    model.eval()
    scores = torch.sigmoid(model(data.edge_index, data.edge_label_index)).cpu().numpy()
    y = data.edge_label.cpu().numpy()
    return roc_auc_score(y, scores), average_precision_score(y, scores), scores


def train_one_seed(data: Data, cfg: dict, seed: int, device) -> dict:
    g = cfg["gnn"]
    set_seed(seed)
    split = RandomLinkSplit(num_val=g["val_ratio"], num_test=g["test_ratio"], is_undirected=True,
                            add_negative_train_samples=False)
    train_data, val_data, test_data = (d.to(device) for d in split(data))

    model = DDIGraphModel(data.num_nodes, g["embedding_dim"], g["hidden_channels"],
                          g["out_channels"], g["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=g["learning_rate"], weight_decay=g["weight_decay"])

    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, g["max_epochs"] + 1):
        model.train()
        opt.zero_grad()
        pos = train_data.edge_label_index
        neg = negative_sampling(train_data.edge_index, num_nodes=data.num_nodes, num_neg_samples=pos.size(1))
        pairs = torch.cat([pos, neg], dim=1)
        labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(device)
        loss = F.binary_cross_entropy_with_logits(model(train_data.edge_index, pairs), labels)
        loss.backward()
        opt.step()

        val_auc, val_ap, _ = evaluate(model, val_data)
        if val_auc > best_auc:
            best_auc, best_state, best_epoch, bad = val_auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= g["patience"]:
                break
        if epoch % 25 == 0:
            print(f"  seed {seed} epoch {epoch:3d} loss {loss.item():.4f} val AUROC {val_auc:.4f} AP {val_ap:.4f}")

    model.load_state_dict(best_state)
    val_auc, val_ap, _ = evaluate(model, val_data)
    test_auc, test_ap, test_scores = evaluate(model, test_data)
    y_test = test_data.edge_label.cpu().numpy()

    baselines = {}
    for name, s in heuristic_scores(test_data.edge_index.cpu(), data.num_nodes,
                                    test_data.edge_label_index.cpu()).items():
        baselines[name] = {"auroc": float(roc_auc_score(y_test, s)), "auprc": float(average_precision_score(y_test, s))}

    ci = bootstrap_ci(y_test, test_scores, roc_auc_score, n_iter=1000, seed=seed)
    print(f"  seed {seed}: best epoch {best_epoch}, val AUROC {val_auc:.4f}, test AUROC {test_auc:.4f} "
          f"(95% CI {ci[0]:.4f}-{ci[1]:.4f}), test AUPRC {test_ap:.4f}")

    # Harder protocol: same test positives, degree-matched negatives.
    mp = test_data.edge_index
    pos = test_data.edge_label_index[:, test_data.edge_label == 1]
    neg = degree_matched_negatives(pos.cpu(), mp.cpu(), data.edge_index, data.num_nodes, seed).to(device)
    dm_pairs = torch.cat([pos, neg], dim=1)
    y_dm = np.concatenate([np.ones(pos.size(1)), np.zeros(neg.size(1))])
    s_dm = score_pairs(model, mp, dm_pairs)
    dm = {"gcn": {"auroc": float(roc_auc_score(y_dm, s_dm)), "auprc": float(average_precision_score(y_dm, s_dm)),
                  "auroc_95ci": bootstrap_ci(y_dm, s_dm, roc_auc_score, n_iter=1000, seed=seed)}}
    h_dm = heuristic_scores(mp.cpu(), data.num_nodes, dm_pairs.cpu())
    for name, s in h_dm.items():
        dm[name] = {"auroc": float(roc_auc_score(y_dm, s)), "auprc": float(average_precision_score(y_dm, s))}
    print(f"  seed {seed}: degree-matched test AUROC GCN {dm['gcn']['auroc']:.4f} | "
          + " | ".join(f"{k} {v['auroc']:.4f}" for k, v in dm.items() if k != "gcn"))

    predictions = {
        "random_negatives": {"y": y_test, "gcn": test_scores,
                             "adamic_adar": heuristic_scores(test_data.edge_index.cpu(), data.num_nodes,
                                                             test_data.edge_label_index.cpu())["adamic_adar"]},
        "degree_matched_negatives": {"y": y_dm, "gcn": s_dm, "adamic_adar": h_dm["adamic_adar"]},
    }
    return {
        "predictions": predictions,
        "degree_matched_test": dm,
        "seed": seed,
        "best_epoch": best_epoch,
        "val": {"auroc": float(val_auc), "auprc": float(val_ap)},
        "test": {"auroc": float(test_auc), "auprc": float(test_ap), "auroc_95ci": ci,
                 "n_pos": int(y_test.sum()), "n_neg": int((1 - y_test).sum())},
        "baselines_test": baselines,
        "state_dict": best_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, node_ids = load_graph(resolve(cfg["paths"]["ddi_edges"]))
    print(f"[GNN] graph: {data.num_nodes} drugs, {data.edge_index.size(1) // 2} undirected interactions")

    runs = [train_one_seed(data, cfg, s, device) for s in args.seeds]

    def agg(key, metric):
        vals = [r[key][metric] for r in runs]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "dataset": "Stanford BioSNAP ChCh-Miner (DrugBank-derived DDI network)",
        "num_nodes": data.num_nodes,
        "num_undirected_edges": int(data.edge_index.size(1) // 2),
        "split": {"train": 1 - cfg["gnn"]["val_ratio"] - cfg["gnn"]["test_ratio"],
                  "val": cfg["gnn"]["val_ratio"], "test": cfg["gnn"]["test_ratio"]},
        "negatives": "random non-edges, 1:1 with positives",
        "seeds": args.seeds,
        "gnn_test": {"auroc": agg("test", "auroc"), "auprc": agg("test", "auprc")},
        "baselines_test": {
            name: {m: {"mean": float(np.mean([r["baselines_test"][name][m] for r in runs])),
                       "std": float(np.std([r["baselines_test"][name][m] for r in runs]))}
                   for m in ("auroc", "auprc")}
            for name in runs[0]["baselines_test"]
        },
        "degree_matched_test": {
            "negatives": "same test positives; one endpoint replaced by a drug sampled proportionally to degree",
            **{name: {m: {"mean": float(np.mean([r["degree_matched_test"][name][m] for r in runs])),
                          "std": float(np.std([r["degree_matched_test"][name][m] for r in runs]))}
                      for m in ("auroc", "auprc")}
               for name in runs[0]["degree_matched_test"]},
        },
        "runs": [{k: v for k, v in r.items() if k not in ("state_dict", "predictions")} for r in runs],
        "config": cfg["gnn"],
    }
    results_dir = resolve(cfg["paths"]["results_dir"])
    save_json(summary, results_dir / "gnn_metrics.json")

    # Test-set scores of the first seed, used for the ROC/PR figures.
    rows = []
    for setting, p in runs[0]["predictions"].items():
        rows.append(pd.DataFrame({"setting": setting, "y": p["y"].astype(int),
                                  "gcn": p["gcn"], "adamic_adar": p["adamic_adar"]}))
    pd.concat(rows).to_csv(results_dir / "gnn_test_predictions.csv", index=False, float_format="%.5f")

    # Keep the first seed's model for downstream use. Embeddings for downstream features are
    # computed with message passing over ALL known edges (no evaluation happens downstream on DDI).
    g = cfg["gnn"]
    model = DDIGraphModel(data.num_nodes, g["embedding_dim"], g["hidden_channels"], g["out_channels"], g["dropout"])
    model.load_state_dict(runs[0]["state_dict"])
    model.eval()
    with torch.no_grad():
        z = model.encode(data.edge_index)
    models_dir = resolve(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": runs[0]["state_dict"], "node_ids": node_ids, "embeddings": z,
                "config": g, "seed": runs[0]["seed"]}, models_dir / "gnn_ddi.pt")

    print("\n[GNN] test AUROC {:.4f} ± {:.4f} | AUPRC {:.4f} ± {:.4f} over seeds {}".format(
        summary["gnn_test"]["auroc"]["mean"], summary["gnn_test"]["auroc"]["std"],
        summary["gnn_test"]["auprc"]["mean"], summary["gnn_test"]["auprc"]["std"], args.seeds))
    for name, m in summary["baselines_test"].items():
        print(f"      baseline {name:24s} AUROC {m['auroc']['mean']:.4f} | AUPRC {m['auprc']['mean']:.4f}")
    print("[GNN] degree-matched negatives (harder):")
    for name, m in summary["degree_matched_test"].items():
        if name != "negatives":
            print(f"      {name:32s} AUROC {m['auroc']['mean']:.4f} ± {m['auroc']['std']:.4f} | "
                  f"AUPRC {m['auprc']['mean']:.4f}")
    print(json.dumps({"saved": str(models_dir / "gnn_ddi.pt")}))

    try:
        from src.utils.plots import make_all

        for f in make_all(cfg):
            print(f"[plots] wrote {f}")
    except ImportError:
        print("[plots] matplotlib not installed - skipping figures")


if __name__ == "__main__":
    main()
