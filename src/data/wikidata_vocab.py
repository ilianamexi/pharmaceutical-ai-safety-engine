"""Build a DrugBank-ID vocabulary (ID, name, synonyms, UNII) from Wikidata (CC0).

Used instead of the official DrugBank Vocabulary CSV when that download is unavailable.
Wikidata property P715 = DrugBank ID; P652 = UNII (FDA Unique Ingredient Identifier).
The output CSV uses the same column names as the DrugBank Vocabulary file, so
src/data/drug_mapping.py reads either one.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "pharmaceutical-ai-safety-engine/0.3 (research portfolio project; python urllib)"

QUERY_MAIN = """
SELECT ?item ?db ?label ?unii WHERE {
  ?item wdt:P715 ?db .
  OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label) = "en") }
  OPTIONAL { ?item wdt:P652 ?unii . }
}"""

QUERY_SYNONYMS = """
SELECT ?item ?alt WHERE {
  ?item wdt:P715 ?db .
  ?item skos:altLabel ?alt . FILTER(LANG(?alt) = "en")
}"""


def normalize_drugbank_id(value: str) -> str | None:
    """Accepts 'DB00945' or '00945' (both spellings occur) and returns 'DB00945'."""
    v = str(value).strip().upper()
    if v.startswith("DB"):
        v = v[2:]
    return f"DB{v}" if v.isdigit() and len(v) == 5 else None


def run_query(query: str, retries: int = 4) -> list[dict]:
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    req = urllib.request.Request(ENDPOINT, data=data, headers={
        "User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))["results"]["bindings"]
        except Exception:  # noqa: BLE001 - network / timeout / 429: back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return []


def build_vocabulary(main_rows: list[dict], synonym_rows: list[dict]) -> list[dict]:
    item_to_db: dict[str, set[str]] = defaultdict(set)
    names: dict[str, str] = {}
    uniis: dict[str, set[str]] = defaultdict(set)
    syns: dict[str, set[str]] = defaultdict(set)

    for r in main_rows:
        item = r["item"]["value"]
        db = normalize_drugbank_id(r["db"]["value"])
        if not db:
            continue
        item_to_db[item].add(db)
        if "label" in r:
            names.setdefault(db, r["label"]["value"])
        if "unii" in r:
            uniis[db].add(r["unii"]["value"].strip().upper())
    for r in synonym_rows:
        for db in item_to_db.get(r["item"]["value"], ()):
            syns[db].add(r["alt"]["value"].strip())

    rows = []
    for db in sorted({d for ds in item_to_db.values() for d in ds}):
        name = names.get(db, "")
        rows.append({
            "DrugBank ID": db,
            "Common name": name,
            "UNII": " | ".join(sorted(uniis[db])),
            "Synonyms": " | ".join(sorted(s for s in syns[db] if s.lower() != name.lower())),
        })
    return rows


def download_wikidata_vocabulary(out_csv: str | Path) -> int:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print("[Wikidata] querying DrugBank IDs, names and UNII codes ...")
    main_rows = run_query(QUERY_MAIN)
    print("[Wikidata] querying English synonyms ...")
    try:
        synonym_rows = run_query(QUERY_SYNONYMS)
    except Exception as e:  # noqa: BLE001 - synonyms improve coverage but are not essential
        print(f"[Wikidata] synonym query failed ({e}); continuing with names only")
        synonym_rows = []
    rows = build_vocabulary(main_rows, synonym_rows)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["DrugBank ID", "Common name", "UNII", "Synonyms"])
        w.writeheader()
        w.writerows(rows)
    n_unii = sum(bool(r["UNII"]) for r in rows)
    print(f"[Wikidata] {len(rows)} DrugBank IDs ({n_unii} with UNII) -> {out_csv}")
    return len(rows)
