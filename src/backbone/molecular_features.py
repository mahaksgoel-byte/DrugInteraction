"""Build Morgan fingerprints and optional external single-drug side-effect features."""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[2]


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def build_fingerprints(drug_table: Path, output: Path, radius: int = 2, bits: int = 2048,
                       side_effect_file: Path | None = None, max_side_effects: int = 1024,
                       pharmgkb_annotations_file: Path | None = None, max_pgx_genes: int = 512) -> None:
    drugs = pd.read_csv(drug_table, dtype=str)
    if not {"drug_id", "canonical_smiles"}.issubset(drugs.columns):
        raise ValueError("drug table must include drug_id and canonical_smiles")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    ids, vectors = [], []
    for row in drugs.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES for {row.drug_id}")
        arr = np.zeros(bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
        ids.append(row.drug_id)
        vectors.append(arr)
    fingerprints = np.stack(vectors)
    arrays = {"drug_ids": np.asarray(ids), "fingerprints": fingerprints,
              "drug_features": fingerprints.astype(np.float32),
              "feature_names": np.asarray([f"morgan_{i}" for i in range(bits)])}
    sider_with_presence = np.zeros((len(ids), 0), dtype=np.uint8)
    sider_names = np.asarray([], dtype=str)
    if side_effect_file is not None and side_effect_file.exists():
        edges = pd.read_csv(side_effect_file, dtype=str)
        required = {"drug_id", "meddra_id", "side_effect_name"}
        if not required.issubset(edges.columns):
            raise ValueError(f"Side-effect table missing columns: {sorted(required - set(edges.columns))}")
        # Prefer standardized MedDRA IDs; use the term name only where ID is absent.
        edges["feature_id"] = edges["meddra_id"].fillna("").str.strip()
        missing = edges["feature_id"].eq("")
        edges.loc[missing, "feature_id"] = "NAME:" + edges.loc[missing, "side_effect_name"].fillna("").str.strip()
        edges = edges[(edges["feature_id"] != "") & edges["drug_id"].isin(ids)].drop_duplicates(["drug_id", "feature_id"])
        prevalence = edges.groupby("feature_id")["drug_id"].nunique().sort_values(ascending=False)
        terms = sorted(prevalence.head(max_side_effects).index.tolist())
        term_index = {term: i for i, term in enumerate(terms)}
        id_index = {drug_id: i for i, drug_id in enumerate(ids)}
        side = np.zeros((len(ids), len(terms)), dtype=np.uint8)
        for drug_id, term in edges[["drug_id", "feature_id"]].itertuples(index=False, name=None):
            col = term_index.get(term)
            if col is not None:
                side[id_index[drug_id], col] = 1
        # Mark source availability from any mapped SIDER record, even when all
        # of that drug's terms fall outside the selected common-term vocabulary.
        has_sider = set(edges["drug_id"].astype(str))
        side_available = np.asarray([[int(drug_id in has_sider)] for drug_id in ids], dtype=np.uint8)
        side_names = np.asarray([f"sider_{x}" for x in terms] + ["sider_source_available"])
        side_with_presence = np.concatenate([side, side_available], axis=1)
        sider_with_presence, sider_names = side_with_presence, side_names
        sider_only_output = output.with_name(output.stem + "_sider_only" + output.suffix)
        np.savez_compressed(sider_only_output, drug_ids=np.asarray(ids), fingerprints=side_with_presence,
                            drug_features=side_with_presence.astype(np.float32), feature_names=side_names)
        print(f"Saved SIDER-only ablation features to {sider_only_output}")
        print(f"Added {len(terms)} SIDER features across {int((side.sum(axis=1) > 0).sum())} drugs")

    morgan_sider = np.concatenate([fingerprints, sider_with_presence], axis=1).astype(np.float32)
    morgan_sider_names = np.concatenate([arrays["feature_names"], sider_names])
    morgan_sider_output = output.with_name(output.stem + "_morgan_sider" + output.suffix)
    np.savez_compressed(morgan_sider_output, drug_ids=np.asarray(ids), fingerprints=fingerprints,
                        drug_features=morgan_sider, feature_names=morgan_sider_names)

    pgx_features = np.zeros((len(ids), 0), dtype=np.uint8)
    pgx_names = np.asarray([], dtype=str)
    if pharmgkb_annotations_file is not None and pharmgkb_annotations_file.exists():
        # PharmGKB annotations identify a drug by name, so only accept exact
        # normalized aliases that resolve to one canonical drug. Ambiguous or
        # combination-drug strings are excluded rather than guessed.
        alias_to_ids: dict[str, set[str]] = {}
        for row in drugs.itertuples(index=False):
            for alias in str(getattr(row, "drug_names", "")).split("|"):
                key = _norm_name(alias)
                if key:
                    alias_to_ids.setdefault(key, set()).add(str(row.drug_id))
        with zipfile.ZipFile(pharmgkb_annotations_file) as archive:
            member = next((name for name in archive.namelist() if name.endswith("clinical_annotations.tsv")), None)
            if member is None:
                raise ValueError(f"No clinical_annotations.tsv found in {pharmgkb_annotations_file}")
            annotations = pd.read_csv(archive.open(member), sep="\t", dtype=str, keep_default_na=False)
        id_index = {drug_id: i for i, drug_id in enumerate(ids)}
        drug_genes: dict[str, set[str]] = {}
        drug_categories: dict[str, set[str]] = {}
        for row in annotations[["Drug(s)", "Gene", "Phenotype Category"]].itertuples(index=False, name=None):
            name, gene_value, category_value = row
            candidates = alias_to_ids.get(_norm_name(name), set())
            if len(candidates) != 1:
                continue
            drug_id = next(iter(candidates))
            genes = {x.strip() for x in re.split(r"[;,|]", gene_value) if x.strip()}
            categories = {x.strip() for x in re.split(r"[;,|]", category_value) if x.strip()}
            drug_genes.setdefault(drug_id, set()).update(genes)
            drug_categories.setdefault(drug_id, set()).update(categories)
        gene_prevalence: dict[str, int] = {}
        for genes in drug_genes.values():
            for gene in genes:
                gene_prevalence[gene] = gene_prevalence.get(gene, 0) + 1
        top_genes = sorted(gene_prevalence, key=lambda gene: (-gene_prevalence[gene], gene))[:max_pgx_genes]
        categories = sorted({c for values in drug_categories.values() for c in values})
        pgx_names = np.asarray([f"pharmgkb_gene_{gene}" for gene in top_genes]
                               + [f"pharmgkb_category_{category}" for category in categories]
                               + ["pharmgkb_source_available"])
        pgx_features = np.zeros((len(ids), len(pgx_names)), dtype=np.uint8)
        gene_index = {gene: i for i, gene in enumerate(top_genes)}
        category_index = {category: len(top_genes) + i for i, category in enumerate(categories)}
        presence_col = len(pgx_names) - 1
        for drug_id in set(drug_genes) | set(drug_categories):
            i = id_index[drug_id]
            for gene in drug_genes.get(drug_id, set()):
                if gene in gene_index:
                    pgx_features[i, gene_index[gene]] = 1
            for category in drug_categories.get(drug_id, set()):
                pgx_features[i, category_index[category]] = 1
            pgx_features[i, presence_col] = 1
        pgx_only_output = output.with_name(output.stem + "_pharmgkb_only" + output.suffix)
        np.savez_compressed(pgx_only_output, drug_ids=np.asarray(ids), fingerprints=pgx_features,
                            drug_features=pgx_features.astype(np.float32), feature_names=pgx_names)
        print(f"Saved PharmGKB-only ablation features to {pgx_only_output}")
        print(f"Added {len(top_genes)} gene and {len(categories)} clinical-category indicators across {len(drug_genes)} exactly mapped drugs")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays["drug_features"] = np.concatenate([morgan_sider, pgx_features.astype(np.float32)], axis=1)
    arrays["feature_names"] = np.concatenate([morgan_sider_names, pgx_names])
    np.savez_compressed(output, **arrays)
    structure_only_output = output.with_name(output.stem + "_structure_only" + output.suffix)
    np.savez_compressed(structure_only_output, drug_ids=np.asarray(ids), fingerprints=fingerprints,
                        drug_features=fingerprints.astype(np.float32),
                        feature_names=np.asarray([f"morgan_{i}" for i in range(bits)]))
    print(f"Saved {len(ids)} drugs with {arrays['drug_features'].shape[1]} total features to {output}")
    print(f"Saved Morgan-only ablation features to {structure_only_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug-table", type=Path, default=ROOT / "data/processed/drug_table.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "cache/features/morgan_r2_2048.npz")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--bits", type=int, default=2048)
    parser.add_argument("--side-effect-file", type=Path, default=ROOT / "data/processed/kg_edges/drug_side_effects.csv.gz")
    parser.add_argument("--max-side-effects", type=int, default=1024)
    parser.add_argument("--pharmgkb-annotations", type=Path,
                        default=ROOT / "data/raw/pharmgkb/clinicalAnnotations.zip")
    parser.add_argument("--max-pgx-genes", type=int, default=512)
    args = parser.parse_args()
    build_fingerprints(args.drug_table, args.output, args.radius, args.bits,
                       args.side_effect_file, args.max_side_effects,
                       args.pharmgkb_annotations, args.max_pgx_genes)


if __name__ == "__main__":
    main()
