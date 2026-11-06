# PharmAI Safety Engine

**Drug-interaction graph learning + adverse-event seriousness prediction on public pharmacovigilance data**

A research / portfolio project built on public drug-safety data (openFDA FAERS, Stanford BioSNAP, Wikidata/DrugBank IDs),
with three parts:

1. **Drug–drug interaction (DDI) link prediction** – a Graph Convolutional Network (PyTorch Geometric)
   trained on the Stanford BioSNAP *ChCh-Miner* network (1,514 DrugBank drugs, 48,514 known interactions).
2. **Adverse-event seriousness prediction** – a classifier trained on real **openFDA FAERS** reports
   to predict the reporter-assigned *serious* flag, using drugs, MedDRA reaction terms, demographics and
   drug-graph features from the GNN. Focal loss is compared against BCE and a logistic-regression baseline.
3. **Case-intake triage demo** – ranks incoming reports by predicted seriousness and shows, for each report,
   the drug pair with the highest predicted interaction probability and whether it is a known interaction.

> ⚠️ **Not for clinical or regulatory use.** FAERS reports do not establish that a drug caused an event,
> contain duplicates and are subject to reporting bias. This project is an engineering and methods exercise.

---

## Results

### 1. DDI link prediction (BioSNAP ChCh-Miner)

Edges split 80 / 10 / 10 (train / val / test) with `RandomLinkSplit`; validation and test edges are hidden
from message passing during training. Negatives are random non-interacting drug pairs (1:1), so a random
scorer has AUROC ≈ AUPRC ≈ 0.5. Mean ± std over 3 seeds (42, 43, 44); full output in
[`results/gnn_metrics.json`](results/gnn_metrics.json).

| Method (test set)                     | AUROC             | AUPRC             |
|---------------------------------------|-------------------|-------------------|
| **GCN (2 layers, learnable embeddings)** | **0.959 ± 0.002** | **0.957 ± 0.003** |
| Adamic–Adar                           | 0.911             | 0.900             |
| Common neighbours                     | 0.909             | 0.895             |
| Preferential attachment (degree product) | 0.882          | 0.866             |

Per-seed 95% bootstrap CI for the GCN test AUROC: 0.957–0.965 (seed 42).
The heuristic baselines are strong because random negatives are "easy" and the network is dense
around high-degree drugs. The GCN improves on them by about 4.8 AUROC points
(0.959 − 0.911 = 0.048).

### 2. Adverse-event seriousness prediction (openFDA FAERS)

10,000 de-duplicated FAERS reports received 2023-01-01 to 2024-12-31. The target is the reporter-assigned
`serious` flag (38.2% positive). Stratified split: 7,000 train / 1,500 validation / 1,500 test.
The 1,040 features are 500 drugs, 500 MedDRA reactions, 4 demographic features and 36 GNN drug-graph features.
Full output: [`results/risk_model_metrics.json`](results/risk_model_metrics.json).

| Model (test set, n = 1,500)         | AUROC [95% CI]          | AUPRC [95% CI]          | Precision | Recall | F1    |
|-------------------------------------|-------------------------|-------------------------|-----------|--------|-------|
| Logistic regression (class-balanced)| 0.943 [0.931–0.954]     | 0.923 [0.906–0.938]     | 0.854     | 0.818  | 0.836 |
| MLP + BCE                           | 0.942 [0.930–0.954]     | 0.925 [0.908–0.940]     | 0.877     | 0.799  | 0.836 |
| MLP + focal loss (α = 0.25, γ = 1)  | 0.941 [0.928–0.953]     | 0.924 [0.907–0.939]     | 0.844     | 0.834  | 0.839 |

Precision, recall and F1 use the threshold that maximises F1 on the validation set.

**Drug mapping coverage:** 22,499 of 24,287 drug mentions (92.6%) were mapped to a DrugBank ID, and
17,969 (74.0%) were also nodes of the BioSNAP graph.

**What the results show**

- **All three models perform about the same.** Their confidence intervals overlap, so the MLPs add
  nothing measurable over logistic regression on these features.
- **Focal loss gives no gain over BCE.** The classes are only moderately imbalanced (38% positive), and
  focal loss is designed for much more extreme imbalance.
- **The GNN drug-graph features do not improve seriousness prediction.** Without them, test AUROC is
  0.942 / 0.942 / 0.943 (LogReg / BCE / focal), versus 0.943 / 0.942 / 0.941 with them, a difference
  far inside the confidence intervals. Seriousness is
  explained mostly by the reported reactions. The GNN remains useful for its own task (DDI prediction,
  above) and as an explanation column in the triage output.

### 3. Check on later data (temporal)

The selected model (MLP + focal) was applied to 200 reports received on 2025-01-01. None of these
reports were used for training, and the threshold was the one fixed on validation (0.222).
Output: [`results/temporal_check.json`](results/temporal_check.json).

| Data                          | n   | % serious | AUROC [95% CI]      | AUPRC | Precision | Recall |
|-------------------------------|-----|-----------|---------------------|-------|-----------|--------|
| Jan 2025 reports (unseen)     | 200 | 45.5%     | 0.860 [0.797–0.906] | 0.865 | 0.661     | 0.835  |

Performance drops from 0.94 to 0.86 AUROC on later data. Part of the drop may be temporal drift, and
the sample is small and comes from a single day. A larger multi-month evaluation is the main next step.

---

## How it works

```text
 Stanford BioSNAP ChCh-Miner                    openFDA FAERS reports (REST API)
 (DrugBank IDs, 48,514 DDI edges)               drugs · MedDRA reactions · age/sex · serious flag
            │                                                 │
            ▼                                                 ▼
 ┌──────────────────────────┐    DrugBank vocabulary   ┌───────────────────────────┐
 │ GCN link predictor (PyG) │◄──── name / UNII ───────►│ Drug mapping + features   │
 │ train / val / test edges │       mapping            │ multi-hot drugs/reactions │
 └────────────┬─────────────┘                          │ demographics              │
              │ drug embeddings, pair DDI probability  │ (+ optional BioBERT text) │
              └────────────────────────┬───────────────┴─────────────┬─────────────┘
                                       ▼                             │
                         ┌────────────────────────────┐              │
                         │ Seriousness classifier      │◄─────────────┘
                         │ LogReg · MLP+BCE · MLP+Focal│  selected on validation AUPRC
                         └─────────────┬──────────────┘
                                       ▼
                         Triage table: p(serious), priority, top DDI pair (known / predicted)
```

**Design decisions**

- **Structured fields instead of NER.** Public FAERS data has no free-text narratives, so entities come
  directly from the structured `drug` and `reaction` fields. This avoids the noise of running an
  NER model on text that would have to be generated from those same fields.
- **Optional BioBERT features.** `src/nlp/biobert_embedder.py` produces mean-pooled embeddings with
  `dmis-lab/biobert-v1.1` from a templated summary of each report. They are off by default because the
  text carries the same information as the structured features; enable with `features: [..., "text"]`.
- **No leakage by construction.** Vocabularies, hyper-parameters (focal α/γ grid), early stopping,
  model selection and the decision threshold use only the training and validation splits. For the
  `death` / `hospitalization` targets, reaction terms that state the outcome (e.g. "death") are removed.
- **Focal loss** (Lin et al., 2017) is implemented with per-class α weighting and unit-tested
  against BCE (γ = 0).

---

## Quick start

Requires Python 3.10+. Exact numbers can differ slightly across PyTorch versions and hardware.

```bash
git clone https://github.com/ilianamexi/pharmaceutical-ai-safety-engine.git
cd pharmaceutical-ai-safety-engine
pip install -r requirements.txt

# 1. Download data: BioSNAP DDI network, ~10,000 openFDA reports (2023–2024) and a
#    DrugBank-ID vocabulary (names, synonyms, UNII) built from Wikidata (CC0)
python -m src.data.download

# 2. Train + evaluate the DDI GNN (≈2 min on CPU)
python -m src.gnn.train_ddi_gnn

# 3. Train + evaluate the seriousness models
python -m src.models.train_risk_model

# 4. Triage NEW reports from a period the models never saw
python -m src.pipeline --fetch 20250101 20250131 --n 200

# Tests
python -m pytest -q
```

All settings live in [`config/config.yaml`](config/config.yaml).
The openFDA API requires a free API key (<https://open.fda.gov/apis/authentication/>); set it as the
`OPENFDA_API_KEY` environment variable before step 1. The key is read from the environment and never stored in the repository.

---

## Repository structure

```text
pharmaceutical-ai-safety-engine/
├── config/config.yaml              # all paths and hyper-parameters
├── data/knowledge_graph/           # BioSNAP edges (downloaded)
├── src/
│   ├── data/
│   │   ├── download.py             # BioSNAP + openFDA + drug-vocabulary download
│   │   ├── openfda.py              # FAERS report parsing (drugs, reactions, seriousness, age, sex)
│   │   ├── wikidata_vocab.py       # DrugBank ID / name / synonym / UNII vocabulary from Wikidata
│   │   └── drug_mapping.py         # drug name / UNII -> DrugBank ID
│   ├── gnn/
│   │   ├── ddi_graph_model.py      # GCN encoder + dot-product decoder (PyG)
│   │   └── train_ddi_gnn.py        # link-prediction training, baselines, metrics
│   ├── models/
│   │   ├── focal_risk_engine.py    # focal loss + MLP
│   │   ├── features.py             # feature blocks (fitted on train split only)
│   │   └── train_risk_model.py     # LogReg / BCE / focal comparison, bootstrap CIs, ablation
│   ├── nlp/biobert_embedder.py     # optional BioBERT mean-pooled embeddings
│   └── pipeline.py                 # triage demo on new reports
├── tests/                          # unit + end-to-end tests (synthetic fixtures only)
└── results/                        # metrics JSON (GNN, seriousness model, temporal check)
```

---

## Limitations

- **FAERS is spontaneous-report data**: no causality, duplicate reports, under-reporting and stimulated
  reporting. The `serious` label is the reporter's regulatory classification, not a clinical gold standard.
- **Outcome-like reaction terms.** Some MedDRA terms ("death", "hospitalisation") almost define the
  `serious` flag. They are kept for the `serious` target because they are known at case intake, but they
  make part of the task easy. They are removed automatically for the `death` / `hospitalization` targets.
- **Random negative sampling** in DDI evaluation is optimistic. Harder protocols (degree-matched negatives,
  held-out drugs, time-based splits) would give lower numbers. The model is transductive: it cannot score
  drugs that are not in the BioSNAP graph.
- **BioSNAP ChCh-Miner** is a static DrugBank snapshot. A missing edge means "not recorded", not "safe".
- **Drug mapping** to DrugBank uses exact name/synonym/UNII matching after salt removal, with a
  community-curated (Wikidata) vocabulary. Mentions that
  fail to map (brand names without generic info, misspellings) get no graph features. Coverage is
  reported in the metrics file.
- The openFDA query returns reports in the API's default order, not a random sample of FAERS.

---

## Data sources and references

- **openFDA** drug adverse event API – U.S. Food & Drug Administration, <https://open.fda.gov/apis/drug/event/>
- **BioSNAP** – Zitnik M., Sosič R., Maheshwari S., Leskovec J. *BioSNAP Datasets: Stanford Biomedical
  Network Dataset Collection* (2018), <https://snap.stanford.edu/biodata/> (ChCh-Miner dataset)
- **DrugBank** – Wishart D.S. et al., *DrugBank 5.0*, Nucleic Acids Research 46 (2018).
  DrugBank IDs are mapped to names, synonyms and UNII codes through **Wikidata** (CC0, properties
  P715 and P652). The official CC0 DrugBank Vocabulary CSV can be used instead by saving it as
  `data/external/drugbank_vocabulary.csv`.
- Kipf T.N., Welling M. *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017
- Lin T.-Y. et al. *Focal Loss for Dense Object Detection*, ICCV 2017
- Lee J. et al. *BioBERT: a pre-trained biomedical language representation model*, Bioinformatics 36(4), 2020

## License

MIT – see [LICENSE](LICENSE). Downloaded datasets remain under their own terms.
