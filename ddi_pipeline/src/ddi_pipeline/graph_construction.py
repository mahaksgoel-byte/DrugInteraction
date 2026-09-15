"""Stages 4-5 — Heterogeneous graph construction and node feature engineering.

Node types: Drug, Protein, Enzyme, SideEffect
Edge types: targets (Drug->Protein), metabolized_by (Drug->Enzyme),
            causes (Drug->SideEffect), ppi (Protein<->Protein),
            interacts_with (Drug<->Drug, the prediction target)

Built with PyTorch Geometric's HeteroData, per the spec's library choice.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from .molecular import smiles_to_fingerprint


def _project(mat: np.ndarray, dim: int, seed: int = 42) -> np.ndarray:
    """Deterministic random projection, used as the light-weight stand-in
    for a ChemBERTa embedding lookup in the demo (production would swap
    this call for `chemberta_embed(smiles)`)."""
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(mat.shape[1], dim)).astype(np.float32) / np.sqrt(mat.shape[1])
    return mat @ proj


def build_drug_features(
    kept_drug_ids: list[str],
    chembl_pubchem: pd.DataFrame,
    pharmgkb: pd.DataFrame,
    sider: pd.DataFrame,
    cyp_enzymes: list[str],
    embedding_dim: int,
    morgan_bits: int,
    side_effect_reduced_dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int]]:
    """Stage 5 — Drug node vector = concat of:
    - Morgan fingerprint projected to `embedding_dim` (ChemBERTa stand-in)
    - 12-dim CYP substrate/inhibitor/inducer x 4-enzyme flag vector
    - Side-effect multi-hot (top-k terms), PCA-reduced to `side_effect_reduced_dim`
    """
    id_to_idx = {d: i for i, d in enumerate(kept_drug_ids)}
    smiles_map = dict(zip(chembl_pubchem["drug_id"], chembl_pubchem["smiles"]))

    fps = np.zeros((len(kept_drug_ids), morgan_bits), dtype=np.float32)
    for d, i in id_to_idx.items():
        smi = smiles_map.get(d)
        fp = smiles_to_fingerprint(smi, n_bits=morgan_bits) if smi else None
        if fp is not None:
            fps[i] = fp
    chem_embed = _project(fps, embedding_dim, seed=seed)

    cyp_cols = [f"{e}_{role}" for e in cyp_enzymes for role in ("substrate", "inhibitor", "inducer")]
    pharmgkb_idx = pharmgkb.set_index("drug_id")
    cyp_feats = np.zeros((len(kept_drug_ids), len(cyp_cols)), dtype=np.float32)
    for d, i in id_to_idx.items():
        if d in pharmgkb_idx.index:
            row = pharmgkb_idx.loc[d]
            cyp_feats[i] = [float(row.get(c, 0.0)) for c in cyp_cols]

    side_effect_terms = sorted(sider["side_effect_term"].unique()) if len(sider) else []
    se_index = {t: i for i, t in enumerate(side_effect_terms)}
    se_multihot = np.zeros((len(kept_drug_ids), max(len(se_index), 1)), dtype=np.float32)
    for _, row in sider.iterrows():
        d, term = row["drug_id"], row["side_effect_term"]
        if d in id_to_idx and term in se_index:
            se_multihot[id_to_idx[d], se_index[term]] = 1.0

    if se_multihot.shape[1] > side_effect_reduced_dim and se_multihot.shape[0] > side_effect_reduced_dim:
        se_reduced = PCA(n_components=side_effect_reduced_dim, random_state=seed).fit_transform(se_multihot)
    else:
        pad = np.zeros((se_multihot.shape[0], side_effect_reduced_dim), dtype=np.float32)
        pad[:, : se_multihot.shape[1]] = se_multihot
        se_reduced = pad

    drug_feats = np.concatenate([chem_embed, cyp_feats, se_reduced], axis=1).astype(np.float32)
    return drug_feats, id_to_idx


def build_protein_features(protein_ids: list[str], feature_dim: int, seed: int = 42) -> np.ndarray:
    """Stage 5 — Protein/enzyme node vector: simple descriptor stand-in
    (production would use sequence length + amino-acid composition or a
    pretrained ProtBERT embedding). Deterministic per protein id here."""
    feats = np.zeros((len(protein_ids), feature_dim), dtype=np.float32)
    for i, pid in enumerate(protein_ids):
        rng = np.random.default_rng(abs(hash(pid)) % (2**32))
        feats[i] = rng.normal(size=feature_dim)
    return feats


def build_hetero_graph(
    sources: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[HeteroData, dict[str, int], dict[str, int]]:
    """Stage 4 — assemble the full heterogeneous graph."""
    kept_drug_ids = sources["_kept_drug_ids"]
    fcfg = cfg["features"]
    dcfg = cfg["data"]

    drug_feats, drug_idx = build_drug_features(
        kept_drug_ids,
        sources["chembl_pubchem"],
        sources["pharmgkb"],
        sources["sider"],
        dcfg["cyp_enzymes"],
        fcfg["drug_embedding_dim"],
        fcfg["morgan_bits"],
        fcfg["side_effect_reduced_dim"],
        seed=cfg.get("seed", 42),
    )

    protein_ids = sorted(set(sources["drugbank_targets"]["protein_id"]) if len(sources["drugbank_targets"]) else [])
    protein_idx = {p: i for i, p in enumerate(protein_ids)}
    protein_feats = build_protein_features(protein_ids, fcfg["protein_feature_dim"], seed=cfg.get("seed", 42))

    enzyme_ids = sorted(set(sources["drugbank_enzymes"]["enzyme_id"]) if len(sources["drugbank_enzymes"]) else [])
    enzyme_idx = {e: i for i, e in enumerate(enzyme_ids)}
    enzyme_feats = build_protein_features(enzyme_ids, fcfg["protein_feature_dim"], seed=cfg.get("seed", 42) + 1)

    side_effect_terms = sorted(sources["sider"]["side_effect_term"].unique()) if len(sources["sider"]) else []
    se_idx = {t: i for i, t in enumerate(side_effect_terms)}
    se_feats = build_protein_features(side_effect_terms, fcfg["protein_feature_dim"], seed=cfg.get("seed", 42) + 2)

    data = HeteroData()
    data["drug"].x = torch.tensor(drug_feats, dtype=torch.float32)
    data["protein"].x = torch.tensor(protein_feats, dtype=torch.float32) if len(protein_ids) else torch.zeros((0, fcfg["protein_feature_dim"]))
    data["enzyme"].x = torch.tensor(enzyme_feats, dtype=torch.float32) if len(enzyme_ids) else torch.zeros((0, fcfg["protein_feature_dim"]))
    data["side_effect"].x = torch.tensor(se_feats, dtype=torch.float32) if len(side_effect_terms) else torch.zeros((0, fcfg["protein_feature_dim"]))

    def edges(df, a_col, b_col, a_idx, b_idx):
        pairs = [
            (a_idx[a], b_idx[b])
            for a, b in zip(df[a_col], df[b_col])
            if a in a_idx and b in b_idx
        ]
        if not pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).T

    data["drug", "targets", "protein"].edge_index = edges(
        sources["drugbank_targets"], "drug_id", "protein_id", drug_idx, protein_idx
    )
    data["drug", "metabolized_by", "enzyme"].edge_index = edges(
        sources["drugbank_enzymes"], "drug_id", "enzyme_id", drug_idx, enzyme_idx
    )
    data["drug", "causes", "side_effect"].edge_index = edges(
        sources["sider"], "drug_id", "side_effect_term", drug_idx, se_idx
    )
    data["protein", "ppi", "protein"].edge_index = edges(
        sources["ppi"], "protein_a_id", "protein_b_id", protein_idx, protein_idx
    ) if len(sources.get("ppi", [])) else torch.zeros((2, 0), dtype=torch.long)

    # All Drug->X edges above only ever point away from Drug nodes, so a
    # relation-aware GNN would never update the Drug embeddings it needs
    # for pair pooling. Add reverse edges so information flows back.
    data = ToUndirected(merge=False)(data)

    # interacts_with is the prediction target, kept as a separate label
    # table rather than baked into the message-passing graph for train
    # pairs (to avoid label leakage); see training.py for pair sampling.
    print(
        f"[graph] Built heterograph: {len(drug_idx)} drugs, {len(protein_idx)} proteins, "
        f"{len(enzyme_idx)} enzymes, {len(se_idx)} side effects. Edge counts: "
        f"targets={data['drug','targets','protein'].edge_index.shape[1]}, "
        f"metabolized_by={data['drug','metabolized_by','enzyme'].edge_index.shape[1]}, "
        f"causes={data['drug','causes','side_effect'].edge_index.shape[1]}, "
        f"ppi={data['protein','ppi','protein'].edge_index.shape[1]}"
    )
    return data, drug_idx, protein_idx
