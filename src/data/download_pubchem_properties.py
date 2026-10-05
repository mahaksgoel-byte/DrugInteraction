"""Download PubChem structure properties for DDInter's cross-referenced CIDs."""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DDINTER = RAW / "ddinter"
OUT = RAW / "pubchem"
BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"


def load_ddinter_ids() -> set[int]:
    ids: set[int] = set()
    for path in DDINTER.glob("ddinter_downloads_code_*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                for col in ("DDInterID_A", "DDInterID_B"):
                    ids.add(int(row[col].removeprefix("DDInter")))
    return ids


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ddinter_ids = load_ddinter_ids()
    xwalk = json.loads((DDINTER / "drug_data.json").read_text(encoding="utf-8"))["records"]
    relevant = [row for row in xwalk if int(row["ddinter"]) in ddinter_ids]
    cids = {
        int(row["pubchem"])
        for row in relevant
        if str(row.get("pubchem", "")).isdigit()
    }
    chembl_ids = {row.get("chembl") for row in relevant if row.get("chembl")}
    mapping_path = RAW / "unichem" / "src1src22.txt.gz"
    import gzip

    with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            chembl, pubchem = line.rstrip("\n").split("\t")[:2]
            if chembl in chembl_ids:
                cids.add(int(pubchem))
    cids = sorted(cids)

    chunks = [cids[i : i + 100] for i in range(0, len(cids), 100)]

    def fetch(i: int, chunk: list[int]) -> tuple[int, dict]:
        path = OUT / f"cid_properties_v2_{i + 1:03d}.json"
        if path.exists() and path.stat().st_size:
            try:
                return i, json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                path.unlink(missing_ok=True)

        url = f"{BASE_URL}/{','.join(map(str, chunk))}/property/Title,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = requests.get(url, timeout=(20, 90), headers={"User-Agent": "DDI-Quantum-Research/0.1 (academic data curation)"})
                response.raise_for_status()
                data = response.json()
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return i, data
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"PubChem request {i + 1} failed: {last_error}")

    # Three workers stay below PubChem's public service request-rate guidance.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, i, chunk) for i, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            i, data = future.result()
            count = len(data.get("PropertyTable", {}).get("Properties", []))
            print(f"Saved PubChem batch {i + 1}/{len(chunks)} ({count} compounds)")

    print(f"Requested {len(cids)} unique PubChem CIDs in {len(chunks)} batches.")


if __name__ == "__main__":
    main()
