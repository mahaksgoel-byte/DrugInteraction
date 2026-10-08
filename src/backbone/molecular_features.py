"""Build structure-only Morgan fingerprints from the processed drug table."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[2]


def build_fingerprints(drug_table: Path, output: Path, radius: int = 2, bits: int = 2048) -> None:
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
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, drug_ids=np.asarray(ids), fingerprints=np.stack(vectors))
    print(f"Saved {len(ids)} fingerprints ({bits} bits, radius {radius}) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug-table", type=Path, default=ROOT / "data/processed/drug_table.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "cache/features/morgan_r2_2048.npz")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--bits", type=int, default=2048)
    args = parser.parse_args()
    build_fingerprints(args.drug_table, args.output, args.radius, args.bits)


if __name__ == "__main__":
    main()
