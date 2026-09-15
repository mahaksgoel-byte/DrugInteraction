"""Stage 1 — Data sources.

Loaders for the four public sources named in the architecture doc:
DrugBank (DDI labels, drug-protein-enzyme relations), ChEMBL/PubChem
(SMILES + bioactivity), PharmGKB (CYP enzyme status), SIDER (side effects).

Real DrugBank and PharmGKB exports require a licensed/free account and a
manual download — this module reads them from `paths.raw_dir` if present
(see docstrings on each loader for the expected filename/columns). If a
source file is missing, `build_synthetic_sources` fabricates a
structurally identical small dataset so the rest of the pipeline
(preprocessing -> graph -> GNN -> heads -> eval) is runnable end to end
without any external download, which is what the hackathon demo runs by
default.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RDKIT_ATOM_VOCAB = ["C", "N", "O", "F", "S", "Cl", "Br", "P"]


def load_drugbank(raw_dir: Path) -> pd.DataFrame | None:
    """Expects `drugbank_ddi.csv` with columns:
    drug_a_id, drug_b_id, interaction (0/1), drug_a_name, drug_b_name.
    Also expects `drugbank_targets.csv` (drug_id, protein_id) and
    `drugbank_enzymes.csv` (drug_id, enzyme_id) alongside it if present.
    """
    f = raw_dir / "drugbank_ddi.csv"
    return pd.read_csv(f) if f.exists() else None


def load_chembl_pubchem(raw_dir: Path) -> pd.DataFrame | None:
    """Expects `chembl_pubchem.csv` with columns: drug_id, smiles, pubchem_cid,
    chembl_id, bioactivity (optional numeric)."""
    f = raw_dir / "chembl_pubchem.csv"
    return pd.read_csv(f) if f.exists() else None


def load_pharmgkb(raw_dir: Path) -> pd.DataFrame | None:
    """Expects `pharmgkb_cyp.csv` with columns: drug_id and one column per
    CYP enzyme x role, e.g. CYP2D6_substrate, CYP2D6_inhibitor, CYP2D6_inducer."""
    f = raw_dir / "pharmgkb_cyp.csv"
    return pd.read_csv(f) if f.exists() else None


def load_sider(raw_dir: Path) -> pd.DataFrame | None:
    """Expects `sider_side_effects.csv` with columns: drug_id, side_effect_term."""
    f = raw_dir / "sider_side_effects.csv"
    return pd.read_csv(f) if f.exists() else None


def _random_smiles(rng: random.Random, min_atoms: int = 4, max_atoms: int = 20) -> str:
    """Fabricate a chemically-plausible-looking SMILES chain for the demo
    dataset. Not real molecules — only used when no raw data is supplied."""
    n = rng.randint(min_atoms, max_atoms)
    atoms = [rng.choice(RDKIT_ATOM_VOCAB) for _ in range(n)]
    chain = atoms[0]
    for a in atoms[1:]:
        bond = rng.choice(["", "", "=", "#"])
        chain += bond + a
    if rng.random() < 0.3:
        chain = "c1ccccc1" + chain  # attach a benzene ring sometimes
    return chain


def build_synthetic_sources(cfg: dict[str, Any], seed: int = 42) -> dict[str, pd.DataFrame]:
    """Builds DrugBank/ChEMBL/PharmGKB/SIDER-shaped tables from scratch so the
    pipeline can run without licensed downloads. Row/column schema matches
    the loaders above exactly, so downstream code never needs to know
    whether data is real or synthetic.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    dcfg = cfg["data"]

    n_drugs = dcfg["synthetic_num_drugs"]
    n_proteins = dcfg["synthetic_num_proteins"]
    n_enzymes = dcfg["synthetic_num_enzymes"]
    n_side_effects = dcfg["synthetic_num_side_effects"]
    cyp_enzymes = dcfg["cyp_enzymes"][:n_enzymes]

    drug_ids = [f"DB{idx:05d}" for idx in range(n_drugs)]
    protein_ids = [f"PR{idx:04d}" for idx in range(n_proteins)]
    side_effect_terms = [f"SE_{idx:03d}" for idx in range(n_side_effects)]

    # ChEMBL/PubChem: SMILES per drug
    chembl_pubchem = pd.DataFrame(
        {
            "drug_id": drug_ids,
            "smiles": [_random_smiles(rng) for _ in drug_ids],
            "pubchem_cid": [f"CID{100000+i}" for i in range(n_drugs)],
            "chembl_id": [f"CHEMBL{200000+i}" for i in range(n_drugs)],
            "bioactivity": np_rng.normal(6.0, 1.5, size=n_drugs).round(2),
        }
    )

    # PharmGKB: CYP substrate/inhibitor/inducer flags
    pharmgkb_rows = {"drug_id": drug_ids}
    for enz in cyp_enzymes:
        for role in ("substrate", "inhibitor", "inducer"):
            pharmgkb_rows[f"{enz}_{role}"] = np_rng.binomial(1, 0.15, size=n_drugs)
    pharmgkb = pd.DataFrame(pharmgkb_rows)

    # SIDER: multiple side-effect rows per drug
    sider_rows = []
    for d in drug_ids:
        k = np_rng.integers(1, min(10, n_side_effects))
        terms = np_rng.choice(side_effect_terms, size=k, replace=False)
        sider_rows.extend({"drug_id": d, "side_effect_term": t} for t in terms)
    sider = pd.DataFrame(sider_rows)

    # DrugBank targets / enzymes (drug -> protein / enzyme edges)
    target_rows = []
    for d in drug_ids:
        k = np_rng.integers(1, 4)
        targets = np_rng.choice(protein_ids, size=k, replace=False)
        target_rows.extend({"drug_id": d, "protein_id": p} for p in targets)
    drugbank_targets = pd.DataFrame(target_rows)

    enzyme_ids = [f"CYP_{e}" for e in cyp_enzymes]
    enzyme_rows = []
    for d in drug_ids:
        k = np_rng.integers(0, min(3, len(enzyme_ids)) + 1)
        if k > 0:
            enzymes = np_rng.choice(enzyme_ids, size=k, replace=False)
            enzyme_rows.extend({"drug_id": d, "enzyme_id": e} for e in enzymes)
    drugbank_enzymes = pd.DataFrame(enzyme_rows)

    # DrugBank DDI labels: positive + negative sampled pairs
    n_pos = n_drugs * 3
    pos_pairs = set()
    while len(pos_pairs) < n_pos:
        a, b = np_rng.choice(n_drugs, size=2, replace=False)
        pos_pairs.add((min(a, b), max(a, b)))
    neg_pairs = set()
    while len(neg_pairs) < n_pos:
        a, b = np_rng.choice(n_drugs, size=2, replace=False)
        pair = (min(a, b), max(a, b))
        if pair not in pos_pairs:
            neg_pairs.add(pair)

    ddi_rows = [
        {"drug_a_id": drug_ids[a], "drug_b_id": drug_ids[b], "interaction": 1}
        for a, b in pos_pairs
    ] + [
        {"drug_a_id": drug_ids[a], "drug_b_id": drug_ids[b], "interaction": 0}
        for a, b in neg_pairs
    ]
    drugbank_ddi = pd.DataFrame(ddi_rows).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Optional protein-protein interaction edges
    ppi_rows = []
    for _ in range(n_proteins):
        a, b = np_rng.choice(n_proteins, size=2, replace=False)
        ppi_rows.append({"protein_a_id": protein_ids[a], "protein_b_id": protein_ids[b]})
    ppi = pd.DataFrame(ppi_rows).drop_duplicates()

    return {
        "drugbank_ddi": drugbank_ddi,
        "drugbank_targets": drugbank_targets,
        "drugbank_enzymes": drugbank_enzymes,
        "chembl_pubchem": chembl_pubchem,
        "pharmgkb": pharmgkb,
        "sider": sider,
        "ppi": ppi,
    }


def load_all_sources(cfg: dict[str, Any], raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Stage 1 entry point: loads every real source it can find and fills
    in the rest with the synthetic fallback if enabled."""
    drugbank_ddi = load_drugbank(raw_dir)
    chembl_pubchem = load_chembl_pubchem(raw_dir)
    pharmgkb = load_pharmgkb(raw_dir)
    sider = load_sider(raw_dir)

    have_all_real = all(x is not None for x in (drugbank_ddi, chembl_pubchem, pharmgkb, sider))
    if have_all_real:
        return {
            "drugbank_ddi": drugbank_ddi,
            "chembl_pubchem": chembl_pubchem,
            "pharmgkb": pharmgkb,
            "sider": sider,
            "drugbank_targets": pd.read_csv(raw_dir / "drugbank_targets.csv")
            if (raw_dir / "drugbank_targets.csv").exists()
            else pd.DataFrame(columns=["drug_id", "protein_id"]),
            "drugbank_enzymes": pd.read_csv(raw_dir / "drugbank_enzymes.csv")
            if (raw_dir / "drugbank_enzymes.csv").exists()
            else pd.DataFrame(columns=["drug_id", "enzyme_id"]),
            "ppi": pd.read_csv(raw_dir / "ppi.csv")
            if (raw_dir / "ppi.csv").exists()
            else pd.DataFrame(columns=["protein_a_id", "protein_b_id"]),
        }

    if not cfg["data"]["use_synthetic_fallback"]:
        missing = [
            n
            for n, v in [
                ("drugbank_ddi.csv", drugbank_ddi),
                ("chembl_pubchem.csv", chembl_pubchem),
                ("pharmgkb_cyp.csv", pharmgkb),
                ("sider_side_effects.csv", sider),
            ]
            if v is None
        ]
        raise FileNotFoundError(
            f"Missing raw source files in {raw_dir}: {missing}. "
            "Either place them there or set data.use_synthetic_fallback: true."
        )

    print(
        "[data_sources] One or more raw source files not found under "
        f"{raw_dir} — generating synthetic fallback dataset for the demo run."
    )
    return build_synthetic_sources(cfg, seed=cfg.get("seed", 42))
