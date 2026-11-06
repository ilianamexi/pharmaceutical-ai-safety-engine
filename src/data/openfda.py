"""openFDA FAERS drug adverse-event reports: download and parsing.

Only *structured* report fields are used (drugs, MedDRA reaction terms, seriousness
flags, age, sex). FAERS public data does not contain free-text clinical narratives.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.fda.gov/drug/event.json"

DRUG_ROLE = {"1": "suspect", "2": "concomitant", "3": "interacting"}
# patientonsetageunit codes -> factor to convert to years
AGE_UNIT_TO_YEARS = {"800": 10.0, "801": 1.0, "802": 1 / 12, "803": 1 / 52, "804": 1 / 365, "805": 1 / 8760}


def _flag(value) -> bool:
    return str(value) == "1"


def parse_report(report: dict) -> dict | None:
    """Turn one raw openFDA report into a flat record. Returns None if unusable."""
    patient = report.get("patient") or {}
    drugs = []
    seen = set()
    for d in patient.get("drug") or []:
        name = (d.get("medicinalproduct") or "").strip()
        active = ((d.get("activesubstance") or {}).get("activesubstancename") or "").strip()
        ofda = d.get("openfda") or {}
        key = (active or name).upper()
        if not key or key in seen:  # drop duplicate drug entries in the same report
            continue
        seen.add(key)
        drugs.append(
            {
                "name": name,
                "active_substance": active,
                "generic_names": ofda.get("generic_name") or [],
                "unii": ofda.get("unii") or [],
                "role": DRUG_ROLE.get(str(d.get("drugcharacterization")), "unknown"),
            }
        )

    reactions = []
    for r in patient.get("reaction") or []:
        pt = (r.get("reactionmeddrapt") or "").strip().lower()
        if pt and pt not in reactions:
            reactions.append(pt)

    serious = report.get("serious")
    if not drugs or not reactions or serious not in ("1", "2", 1, 2):
        return None

    age = None
    try:
        age_val = float(patient.get("patientonsetage"))
        factor = AGE_UNIT_TO_YEARS.get(str(patient.get("patientonsetageunit")))
        if factor is not None:
            age = round(age_val * factor, 2)
            if not 0 <= age <= 120:
                age = None
    except (TypeError, ValueError):
        pass

    sex = {"1": "male", "2": "female"}.get(str(patient.get("patientsex")), "unknown")

    return {
        "safetyreportid": str(report.get("safetyreportid")),
        "receivedate": report.get("receivedate"),
        "serious": _flag(serious),
        "death": _flag(report.get("seriousnessdeath")),
        "hospitalization": _flag(report.get("seriousnesshospitalization")),
        "age_years": age,
        "sex": sex,
        "drugs": drugs,
        "reactions": reactions,
    }


def _get_json(url: str, retries: int = 5) -> dict:
    # A browser-style User-Agent: some API gateways reject unknown client strings with 403.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; pharmaceutical-ai-safety-engine)",
                                               "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:  # openFDA returns 404 when a page has no results
                return {"results": []}
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"openFDA HTTP {e.code} for {url}\nResponse: {body}") from None
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return {"results": []}


def fetch_reports(
    out_path: str | Path,
    date_from: str,
    date_to: str,
    n_reports: int = 10000,
    page_size: int = 1000,
    sleep_seconds: float = 0.5,
    api_key: str | None = None,
) -> int:
    """Download up to n_reports reports received in [date_from, date_to] and write JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not api_key:
        raise RuntimeError(
            "openFDA requires a (free) API key: get one at https://open.fda.gov/apis/authentication/ "
            "and set the OPENFDA_API_KEY environment variable.")
    records: dict[str, dict] = {}
    skip = 0
    while skip < n_reports:
        limit = min(page_size, n_reports - skip)
        # api_key goes first (as in the openFDA docs). Brackets percent-encoded; "+" = space.
        url = (f"{BASE_URL}?api_key={api_key}&search=receivedate:%5B{date_from}+TO+{date_to}%5D"
               f"&limit={limit}&skip={skip}")
        results = _get_json(url).get("results", [])
        if not results:
            break
        for raw in results:
            rec = parse_report(raw)
            if rec:
                records[rec["safetyreportid"]] = rec  # keeps one version per report id
        print(f"[openFDA] fetched {skip + len(results)} reports, {len(records)} usable so far")
        skip += len(results)
        time.sleep(sleep_seconds)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec) + "\n")
    return len(records)


def load_reports(path: str | Path):
    import pandas as pd

    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def api_key_from_env(var: str) -> str | None:
    return os.environ.get(var) or None
