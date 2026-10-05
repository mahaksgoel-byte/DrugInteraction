"""Harmonize and clean DDInter and its independently sourced annotations."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def read_tabular_zip(path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        return pd.read_csv(stream, sep="\t", dtype=str, keep_default_na=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "kg_edges").mkdir(parents=True, exist_ok=True)

    # Merge the eight untouched category downloads.
    raw_pairs: list[dict[str, str]] = []
    categories: dict[str, int] = {}
    for path in sorted((RAW / "ddinter").glob("ddinter_downloads_code_*.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        expected = {"DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"}
        if set(frame.columns) != expected:
            raise ValueError(f"Unexpected DDInter schema in {path.name}: {list(frame.columns)}")
        frame["source_file"] = path.name
        raw_pairs.extend(frame.to_dict("records"))
        categories[path.stem.rsplit("_", 1)[-1]] = len(frame)

    # DDInter's public plugin export contains the external-ID crosswalk; the
    # UniChem source tables provide an independently versioned ID crosswalk.
    drug_ids = json.loads((RAW / "ddinter" / "drug_data.json").read_text(encoding="utf-8"))["records"]
    by_ddinter = {int(row["ddinter"]): row for row in drug_ids}
    chembl_to_cid: dict[str, set[str]] = defaultdict(set)
    with gzip.open(RAW / "unichem" / "src1src22.txt.gz", "rt", encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            chembl, cid = line.rstrip("\n").split("\t")[:2]
            chembl_to_cid[chembl].add(cid)

    properties: dict[str, dict] = {}
    for path in sorted((RAW / "pubchem").glob("cid_properties*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for prop in data.get("PropertyTable", {}).get("Properties", []):
            properties[str(prop["CID"])] = prop

    # PharmGKB/ClinPGx chemical records can fill structure/ID gaps where a
    # generic name matches exactly. Keep the source's own identifier in audit.
    pgx_chemicals = read_tabular_zip(RAW / "pharmgkb" / "chemicals.zip", "chemicals.tsv")
    pgx_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pgx_chemicals.to_dict("records"):
        for name in [row.get("Name", ""), row.get("Generic Names", "")]:
            for alias in re.split(r"[;,|]", name):
                if alias.strip():
                    pgx_by_name[norm_name(alias)].append(row)

    id_name: dict[str, str] = {}
    for row in raw_pairs:
        for id_col, name_col in (("DDInterID_A", "Drug_A"), ("DDInterID_B", "Drug_B")):
            drug_id = row[id_col]
            old_name = id_name.setdefault(drug_id, row[name_col])
            if old_name != row[name_col]:
                raise ValueError(f"Conflicting names for {drug_id}: {old_name!r}, {row[name_col]!r}")

    drug_rows: list[dict] = []
    for ddi_id, name in sorted(id_name.items(), key=lambda x: int(x[0].removeprefix("DDInter"))):
        numeric_id = int(ddi_id.removeprefix("DDInter"))
        ext = by_ddinter.get(numeric_id, {})
        drugbank_id = str(ext.get("drugbank", "")).strip()
        chembl_id = str(ext.get("chembl", "")).strip()
        cid = str(ext.get("pubchem", "")).strip()
        if not cid and chembl_id:
            candidates = sorted(chembl_to_cid.get(chembl_id, set()), key=int)
            cid = candidates[0] if candidates else ""
        pgx = pgx_by_name.get(norm_name(name), [])
        if not cid and pgx:
            pgx_cids = sorted(
                {v.strip() for row in pgx for v in re.split(r"[;,|]", row.get("PubChem Compound Identifiers", "")) if v.strip().isdigit()},
                key=int,
            )
            cid = pgx_cids[0] if pgx_cids else ""
        prop = properties.get(cid, {})
        smiles = prop.get("SMILES", "") or prop.get("ConnectivitySMILES", "")
        valid = False
        canonical_smiles = ""
        reason = ""
        if not cid:
            reason = "no_pubchem_mapping"
        elif not smiles:
            reason = "no_pubchem_smiles"
        else:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                reason = "rdkit_sanitize_failed"
            else:
                valid = True
                canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        drug_rows.append(
            {
                "ddinter_id": ddi_id,
                "drug_name": name,
                "drugbank_id": drugbank_id,
                "chembl_id": chembl_id,
                "pubchem_cid": cid,
                "pubchem_title": prop.get("Title", ""),
                "inchi_key": prop.get("InChIKey", ""),
                "source_smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "rdkit_valid": valid,
                "exclusion_reason": reason,
                "pharmgkb_exact_name_matches": len(pgx),
            }
        )

    all_drugs = pd.DataFrame(drug_rows)
    valid_source_drugs = all_drugs.loc[all_drugs["rdkit_valid"]].copy()
    valid_source_drugs["drug_id"] = valid_source_drugs.apply(
        lambda row: f"INCHIKEY:{row['inchi_key']}" if row["inchi_key"] else f"PUBCHEM:{row['pubchem_cid']}",
        axis=1,
    )
    all_drugs["canonical_drug_id"] = ""
    all_drugs.loc[valid_source_drugs.index, "canonical_drug_id"] = valid_source_drugs["drug_id"]
    all_drugs.to_csv(PROCESSED / "drug_mapping_audit.csv", index=False)
    # InChIKey groups collapse duplicate DDInter entities that resolve to the
    # same chemical. Keep all source identifiers/names as aliases for audit.
    clean_drugs = (
        valid_source_drugs.groupby("drug_id", sort=True, as_index=False)
        .agg(
            ddinter_ids=("ddinter_id", lambda x: "|".join(sorted(set(x), key=lambda y: int(y[7:])))),
            drug_names=("drug_name", lambda x: "|".join(sorted(set(x), key=str.casefold))),
            drugbank_ids=("drugbank_id", lambda x: "|".join(sorted({v for v in x if v}))),
            chembl_ids=("chembl_id", lambda x: "|".join(sorted({v for v in x if v}))),
            pubchem_cids=("pubchem_cid", lambda x: "|".join(sorted({v for v in x if v}, key=int))),
            inchi_key=("inchi_key", "first"),
            canonical_smiles=("canonical_smiles", "first"),
        )
    )
    clean_drugs.to_csv(PROCESSED / "drug_table.csv", index=False)
    ddinter_to_drug = dict(zip(valid_source_drugs["ddinter_id"], valid_source_drugs["drug_id"]))
    valid_ids = set(ddinter_to_drug)

    # Pair order is canonicalized using the internal DDInter IDs. Duplicates
    # retain every reported severity label while remaining one binary positive.
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_pair_count = 0
    self_pair_count = 0
    for row in raw_pairs:
        a, b = sorted((row["DDInterID_A"], row["DDInterID_B"]), key=lambda x: int(x.removeprefix("DDInter")))
        if a == b:
            self_pair_count += 1
            continue
        grouped[(a, b)].add(row["Level"].strip())
        raw_pair_count += 1
    all_pair_rows = [
        {"ddinter_id_a": a, "ddinter_id_b": b, "severity_levels": "|".join(sorted(levels)), "label": 1}
        for (a, b), levels in sorted(grouped.items(), key=lambda x: (int(x[0][0][7:]), int(x[0][1][7:])))
    ]
    all_pairs = pd.DataFrame(all_pair_rows)
    all_pairs.to_csv(PROCESSED / "ddi_positive_pairs_all.csv", index=False)
    clean_grouped: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: {"levels": set(), "a_aliases": set(), "b_aliases": set()})
    unmapped_pair_count = 0
    collapsed_self_pair_count = 0
    missing_source_pairs: set[tuple[str, str]] = set()
    collapsed_self_source_pairs: set[tuple[str, str]] = set()
    for row in raw_pairs:
        a, b = row["DDInterID_A"], row["DDInterID_B"]
        if a not in ddinter_to_drug or b not in ddinter_to_drug:
            unmapped_pair_count += 1
            missing_source_pairs.add(tuple(sorted((a, b), key=lambda x: int(x.removeprefix("DDInter")))))
            continue
        drug_a, drug_b = ddinter_to_drug[a], ddinter_to_drug[b]
        if drug_a == drug_b:
            collapsed_self_pair_count += 1
            collapsed_self_source_pairs.add(tuple(sorted((a, b), key=lambda x: int(x.removeprefix("DDInter")))))
            continue
        if drug_a < drug_b:
            key, alias_a, alias_b = (drug_a, drug_b), a, b
        else:
            key, alias_a, alias_b = (drug_b, drug_a), b, a
        clean_grouped[key]["levels"].add(row["Level"].strip())
        clean_grouped[key]["a_aliases"].add(alias_a)
        clean_grouped[key]["b_aliases"].add(alias_b)
        clean_grouped[key].setdefault("source_pairs", set()).add(
            tuple(sorted((a, b), key=lambda x: int(x.removeprefix("DDInter"))))
        )
    clean_pair_rows = []
    for (drug_a, drug_b), values in sorted(clean_grouped.items()):
        clean_pair_rows.append(
            {
                "drug_id_a": drug_a,
                "drug_id_b": drug_b,
                "ddinter_ids_a": "|".join(sorted(values["a_aliases"], key=lambda y: int(y[7:]))),
                "ddinter_ids_b": "|".join(sorted(values["b_aliases"], key=lambda y: int(y[7:]))),
                "severity_levels": "|".join(sorted(values["levels"])),
                "label": 1,
            }
        )
    clean_pairs = pd.DataFrame(clean_pair_rows)
    clean_pairs.to_csv(PROCESSED / "ddi_positive_pairs.csv", index=False)

    # Independent SIDER adverse-effect edges, keyed by PubChem CID. The input
    # columns have no header; the first STITCH compound identifier is CID-based.
    cid_to_drugs: dict[str, set[str]] = defaultdict(set)
    for row in valid_source_drugs.to_dict("records"):
        if row["pubchem_cid"]:
            cid_to_drugs[str(row["pubchem_cid"])].add(row["drug_id"])
    side_effect_rows: list[tuple[str, str, str, str]] = []
    with gzip.open(RAW / "sider" / "meddra_all_se.tsv.gz", "rt", encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            # SIDER stores both STITCH flat and stereo identifiers. The
            # second column contains the PubChem stereo CID cross-reference.
            cid_match = re.fullmatch(r"CID0*(\d+)", row[1])
            if not cid_match:
                continue
            cid = str(int(cid_match.group(1)))
            for drug_id in cid_to_drugs.get(cid, set()):
                side_effect_rows.append((drug_id, cid, row[4], row[5]))
    side_effects = pd.DataFrame(
        sorted(set(side_effect_rows)),
        columns=["drug_id", "pubchem_cid", "meddra_id", "side_effect_name"],
    )
    side_effects.to_csv(PROCESSED / "kg_edges" / "drug_side_effects.csv.gz", index=False, compression="gzip")

    # Preserve full human-only STRING download; its required score threshold
    # is applied here, and each undirected association appears once.
    ppi: dict[tuple[str, str], int] = {}
    with gzip.open(RAW / "string" / "9606.protein.links.v12.5.txt.gz", "rt", encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            p1, p2, score_text = line.split()
            score = int(score_text)
            if score < 700:
                continue
            a, b = sorted((p1, p2))
            if a != b:
                ppi[(a, b)] = max(score, ppi.get((a, b), 0))
    ppi_frame = pd.DataFrame(
        ((a, b, score) for (a, b), score in sorted(ppi.items())),
        columns=["protein_string_id_a", "protein_string_id_b", "combined_score"],
    )
    ppi_frame.to_csv(PROCESSED / "kg_edges" / "string_human_ppi_score_ge_700.csv.gz", index=False, compression="gzip")

    # Degree and source-coverage summary for this DDInter benchmark.
    degree = Counter()
    for row in clean_pairs.to_dict("records"):
        degree[row["drug_id_a"]] += 1
        degree[row["drug_id_b"]] += 1
    degree_values = sorted(degree.values())
    degree_histogram = Counter(degree_values)
    pd.DataFrame(
        ((value, count) for value, count in sorted(degree_histogram.items())),
        columns=["positive_pair_degree", "drug_count"],
    ).to_csv(PROCESSED / "drug_degree_distribution.csv", index=False)
    levels = Counter(level for row in raw_pairs for level in [row["Level"].strip()])
    stats = {
        "source_scope": "DDInter 2.0 public category downloads; DrugBank label file not included (academic account/licence gated)",
        "ddinter_category_rows": categories,
        "raw_interaction_rows": len(raw_pairs),
        "raw_unique_drugs": len(id_name),
        "id_harmonization_counts": {
            "ddinter_drug_ids": len(id_name),
            "matched_supplemental_ddinter_crosswalk": sum(int(drug_id.removeprefix("DDInter")) in by_ddinter for drug_id in id_name),
            "with_any_drugbank_chembl_or_pubchem_id": int((all_drugs[["drugbank_id", "chembl_id", "pubchem_cid"]].ne("").any(axis=1)).sum()),
            "with_pubchem_cid_after_unichem_and_clinpgx_fallback": int(all_drugs["pubchem_cid"].ne("").sum()),
            "with_pubchem_structure": int(all_drugs["source_smiles"].ne("").sum()),
            "rdkit_sanitized": int(all_drugs["rdkit_valid"].sum()),
            "unique_structures_after_inchikey_deduplication": int(clean_drugs["drug_id"].nunique()),
        },
        "raw_unique_canonical_pairs": len(grouped),
        "duplicate_pair_rows_collapsed": len(raw_pairs) - self_pair_count - len(grouped),
        "self_pair_rows_dropped": self_pair_count,
        "severity_row_counts": dict(levels),
        "source_drug_ids_with_pubchem_structure_and_rdkit_sanitization": len(valid_source_drugs),
        "unique_canonical_drugs_after_structure_deduplication": len(clean_drugs),
        "drugs_excluded_by_reason": all_drugs.loc[~all_drugs["rdkit_valid"], "exclusion_reason"].value_counts().to_dict(),
        "clean_positive_pairs": len(clean_pairs),
        "raw_pair_rows_without_both_valid_structures": unmapped_pair_count,
        "unique_source_pairs_without_both_valid_structures": len(missing_source_pairs),
        "pair_rows_collapsed_to_structure_self_pairs": collapsed_self_pair_count,
        "unique_source_pairs_collapsed_to_structure_self_pairs": len(collapsed_self_source_pairs),
        "unique_source_pairs_merged_by_structure_deduplication": sum(max(0, len(v.get("source_pairs", set())) - 1) for v in clean_grouped.values()),
        "clean_pair_severity_labels": dict(Counter(level for cell in clean_pairs["severity_levels"] for level in cell.split("|"))),
        "clean_ddi_degree_distribution": {
            "drug_count_with_degree": len(degree_values),
            "min": min(degree_values) if degree_values else 0,
            "median": float(pd.Series(degree_values).median()) if degree_values else 0,
            "mean": float(pd.Series(degree_values).mean()) if degree_values else 0,
            "max": max(degree_values) if degree_values else 0,
        },
        "degree_histogram_file": "drug_degree_distribution.csv",
        "sider_independent_drug_side_effect_edges_mapped_to_ddinter": len(side_effects),
        "string_human_ppi_edges_with_score_ge_700": len(ppi_frame),
        "drug_target_and_drug_enzyme_edges": "not generated: DrugBank labels/target/enzyme tables require the academic account-gated package",
        "note": "Unlisted drug pairs are unlabeled, not confirmed negatives. No negatives or splits were generated in this cleaning pass.",
    }
    (PROCESSED / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    pd.DataFrame([{"statistic": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in stats.items()]).to_csv(
        PROCESSED / "dataset_stats.csv", index=False
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
