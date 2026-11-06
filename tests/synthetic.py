"""SYNTHETIC test fixtures only — used to check that the code runs end to end.
Nothing generated here is used for any reported result."""
from __future__ import annotations

import json
import random
from pathlib import Path

# Small vocabulary in the DrugBank Vocabulary CSV layout (test fixture).
VOCAB_ROWS = [
    ("DB00945", "Aspirin", "R16CO5Y76E", "Acetylsalicylic acid | ASA"),
    ("DB00682", "Warfarin", "5Q7ZVV76EI", "Coumadin"),
    ("DB00316", "Acetaminophen", "362O9ITL9D", "Paracetamol"),
    ("DB00331", "Metformin", "9100L32L2N", ""),
    ("DB01050", "Ibuprofen", "WK2XYI10QM", ""),
    ("DB00641", "Simvastatin", "AGG2FN16EV", ""),
    ("DB00254", "Doxycycline", "N12000U13O", ""),
    ("DB00515", "Cisplatin", "Q20Q21Q62J", ""),
]
DRUGS = ["ASPIRIN", "WARFARIN SODIUM", "ACETAMINOPHEN", "METFORMIN HYDROCHLORIDE", "IBUPROFEN",
         "SIMVASTATIN", "DOXYCYCLINE HYCLATE", "CISPLATIN", "UNMAPPED DRUG X"]
REACTIONS = ["nausea", "headache", "haemorrhage", "renal failure", "rash", "death", "dizziness", "hepatotoxicity"]
SEVERE = {"haemorrhage", "renal failure", "death", "hepatotoxicity"}


def write_vocab(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write('"DrugBank ID","Accession Numbers","Common name","CAS","UNII","Synonyms","Standard InChI Key"\n')
        for db, name, unii, syn in VOCAB_ROWS:
            f.write(f'"{db}","","{name}","","{unii}","{syn}",""\n')


def write_reports(path: Path, n: int = 600, seed: int = 0) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            drugs = rng.sample(DRUGS, rng.randint(1, 4))
            reactions = rng.sample(REACTIONS, rng.randint(1, 3))
            serious = bool(SEVERE & set(reactions)) or rng.random() < 0.1
            rec = {
                "safetyreportid": str(100000 + i), "receivedate": "20240101",
                "serious": serious, "death": "death" in reactions, "hospitalization": serious and rng.random() < 0.5,
                "age_years": rng.choice([None, rng.uniform(18, 90)]), "sex": rng.choice(["male", "female", "unknown"]),
                "drugs": [{"name": d, "active_substance": d, "generic_names": [], "unii": [],
                           "role": "suspect" if j == 0 else "concomitant"} for j, d in enumerate(drugs)],
                "reactions": reactions,
            }
            f.write(json.dumps(rec) + "\n")
