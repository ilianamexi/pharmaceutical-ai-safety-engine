"""Download all public datasets used by the project.

Usage (from the project root):
    python -m src.data.download                   # BioSNAP + openFDA + drug vocabulary
    python -m src.data.download --only drugvocab  # DrugBank-ID vocabulary from Wikidata
"""
from __future__ import annotations

import argparse
import gzip
import io
import urllib.request

import pandas as pd

from src.data.openfda import api_key_from_env, fetch_reports
from src.data.wikidata_vocab import download_wikidata_vocabulary
from src.utils.common import load_config, resolve

# Stanford BioSNAP ChCh-Miner: drug-drug interaction network derived from DrugBank
# (the "durgbank" typo is part of the official file name).
BIOSNAP_URL = "https://snap.stanford.edu/biodata/datasets/10001/files/ChCh-Miner_durgbank-chem-chem.tsv.gz"


def download_biosnap(out_csv) -> int:
    out_csv = resolve(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(BIOSNAP_URL, headers={"User-Agent": "pharmaceutical-ai-safety-engine (research)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = gzip.decompress(resp.read()).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["Drug_A_ID", "Drug_B_ID"])
    df.to_csv(out_csv, index=False)
    print(f"[BioSNAP] {len(df)} interaction edges between "
          f"{pd.unique(df.values.ravel()).size} DrugBank drugs -> {out_csv}")
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["biosnap", "openfda", "drugvocab"], default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.only in (None, "biosnap"):
        download_biosnap(cfg["paths"]["ddi_edges"])

    if args.only in (None, "openfda"):
        o = cfg["openfda"]
        try:
            n = fetch_reports(
                resolve(cfg["paths"]["openfda_raw"]),
                date_from=o["date_from"],
                date_to=o["date_to"],
                n_reports=o["n_reports"],
                page_size=o["page_size"],
                sleep_seconds=o["sleep_seconds"],
                api_key=api_key_from_env(o["api_key_env"]),
            )
            print(f"[openFDA] saved {n} de-duplicated reports -> {cfg['paths']['openfda_raw']}")
        except Exception as e:  # noqa: BLE001 - keep going so the other downloads still run
            print(f"[openFDA] download failed: {e}\n"
                  "          Re-run later with: python -m src.data.download --only openfda")

    if args.only in (None, "drugvocab"):
        vocab = resolve(cfg["paths"]["drugbank_vocabulary"])
        if vocab.exists() and args.only is None:
            print(f"[vocab] {vocab} already exists (official DrugBank or Wikidata file) - kept")
        else:
            try:
                download_wikidata_vocabulary(vocab)
            except Exception as e:  # noqa: BLE001
                print(f"[vocab] Wikidata query failed: {e}\n"
                      "        Re-run later with: python -m src.data.download --only drugvocab\n"
                      "        Without this file the 'gnn' features of the risk model are disabled.")


if __name__ == "__main__":
    main()
