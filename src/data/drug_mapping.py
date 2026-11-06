"""Map openFDA drug mentions to DrugBank IDs (the node IDs of the BioSNAP graph).

Reads either the CC0 "DrugBank Vocabulary" CSV (https://go.drugbank.com/releases/latest#open-data)
or the equivalent file built from Wikidata by src/data/wikidata_vocab.py.
Matching order per drug: UNII code -> active substance -> openFDA generic names -> product name,
each tried as-is and with common salt/ester suffixes removed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SALT_WORDS = {
    "HYDROCHLORIDE", "HCL", "SODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM", "SULFATE", "SULPHATE",
    "MESYLATE", "MALEATE", "TARTRATE", "BITARTRATE", "CITRATE", "ACETATE", "BESYLATE", "FUMARATE",
    "SUCCINATE", "PHOSPHATE", "BROMIDE", "CHLORIDE", "HYCLATE", "MONOHYDRATE", "DIHYDRATE",
    "TRIHYDRATE", "HYDRATE", "ANHYDROUS", "DISODIUM", "DIPROPIONATE", "PROPIONATE", "VALERATE",
    "LACTATE", "GLUCONATE", "NITRATE", "OXALATE", "PAMOATE", "TOSYLATE", "ESYLATE", "HYDROBROMIDE",
    "MEGLUMINE", "TROMETHAMINE", "ERBUMINE", "DECANOATE", "ENANTHATE", "CYPIONATE",
}


def normalize_name(name: str) -> str:
    name = re.sub(r"[^A-Z0-9 ]+", " ", str(name).upper())
    return re.sub(r"\s+", " ", name).strip()


def strip_salts(name: str) -> str:
    tokens = [t for t in normalize_name(name).split() if t not in SALT_WORDS]
    return " ".join(tokens)


def _find_col(df: pd.DataFrame, needle: str) -> str | None:
    for c in df.columns:
        if needle in c.lower():
            return c
    return None


class DrugBankMapper:
    def __init__(self, vocabulary_csv: str | Path):
        vocab = pd.read_csv(vocabulary_csv, dtype=str).fillna("")
        id_col = _find_col(vocab, "drugbank id")
        name_col = _find_col(vocab, "common name")
        unii_col = _find_col(vocab, "unii")
        syn_col = _find_col(vocab, "synonym")
        if not id_col or not name_col:
            raise ValueError(f"Unexpected DrugBank vocabulary columns: {list(vocab.columns)}")

        self.by_unii: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        for _, row in vocab.iterrows():
            db_id = row[id_col].strip()
            if unii_col:
                for u in row[unii_col].split("|"):  # Wikidata-built file may list several
                    if u.strip():
                        self.by_unii[u.strip().upper()] = db_id
            names = [row[name_col]]
            if syn_col and row[syn_col]:
                names += row[syn_col].split("|")
            for n in names:
                for key in (normalize_name(n), strip_salts(n)):
                    if key:
                        self.by_name.setdefault(key, db_id)  # first (common name) wins

    def map_name(self, name: str) -> str | None:
        for key in (normalize_name(name), strip_salts(name)):
            if key and key in self.by_name:
                return self.by_name[key]
        return None

    def map_drug(self, drug: dict) -> list[str]:
        """Return the DrugBank IDs for one openFDA drug entry (several for combination products)."""
        ids = [self.by_unii[u.upper()] for u in (drug.get("unii") or []) if u.upper() in self.by_unii]
        if ids:
            return sorted(set(ids))
        candidates = [drug.get("active_substance", "")] + list(drug.get("generic_names") or []) + [drug.get("name", "")]
        for c in candidates:
            # Combination products ("A AND B", "A\B", "A, B") are split into their components.
            parts = [p for p in re.split(r"\\|/|,| AND |;", str(c).upper()) if p.strip()]
            hits = [h for h in (self.map_name(p) for p in parts) if h]
            if hits:
                return sorted(set(hits))
        return []


def load_mapper(path: str | Path) -> DrugBankMapper | None:
    path = Path(path)
    if not path.exists():
        return None
    return DrugBankMapper(path)
