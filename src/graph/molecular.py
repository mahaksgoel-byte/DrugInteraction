"""Serialize RDKit atom/bond graphs for every canonical drug.

The output is a packed NumPy archive. ``node_ptr`` and ``edge_ptr`` delimit the
atom and directed bond rows belonging to each entry in ``drug_ids``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
ATOMIC_NUMBERS = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
HYBRIDIZATIONS = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                  Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                  Chem.HybridizationType.SP3D2)
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)


def atom_features(atom: Chem.Atom) -> list[int]:
    values = [int(atom.GetAtomicNum() == n) for n in ATOMIC_NUMBERS]
    values += [int(atom.GetAtomicNum() not in ATOMIC_NUMBERS)]
    values += [int(atom.GetTotalDegree() == n) for n in range(6)]
    values += [int(atom.GetFormalCharge() == n) for n in range(-2, 3)]
    values += [int(atom.GetIsAromatic())]
    values += [int(atom.GetHybridization() == h) for h in HYBRIDIZATIONS]
    values += [int(atom.GetHybridization() not in HYBRIDIZATIONS)]
    return values


def bond_features(bond: Chem.Bond) -> list[int]:
    values = [int(bond.GetBondType() == kind) for kind in BOND_TYPES]
    values += [int(bond.GetBondType() not in BOND_TYPES), int(bond.IsInRing()), int(bond.GetIsConjugated())]
    return values


def build_graphs(drug_table: Path, output: Path) -> None:
    drugs = pd.read_csv(drug_table, dtype=str)
    atom_rows: list[list[int]] = []
    edge_rows: list[tuple[int, int]] = []
    bond_rows: list[list[int]] = []
    node_ptr, edge_ptr = [0], [0]
    for row in drugs.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES for {row.drug_id}")
        offset = len(atom_rows)
        atom_rows.extend(atom_features(atom) for atom in mol.GetAtoms())
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx() + offset, bond.GetEndAtomIdx() + offset
            features = bond_features(bond)
            edge_rows.extend(((a, b), (b, a)))
            bond_rows.extend((features, features))
        node_ptr.append(len(atom_rows))
        edge_ptr.append(len(edge_rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output,
        drug_ids=drugs["drug_id"].to_numpy(dtype=str),
        atom_features=np.asarray(atom_rows, dtype=np.uint8),
        edge_index=np.asarray(edge_rows, dtype=np.int32).reshape(-1, 2).T,
        bond_features=np.asarray(bond_rows, dtype=np.uint8).reshape(-1, 7),
        node_ptr=np.asarray(node_ptr, dtype=np.int64), edge_ptr=np.asarray(edge_ptr, dtype=np.int64))
    summary = {"drug_count": len(drugs), "atom_count": len(atom_rows), "directed_bond_count": len(edge_rows),
               "atom_feature_dim": len(atom_rows[0]) if atom_rows else 0, "bond_feature_dim": 7}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved packed molecular graphs: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug-table", type=Path, default=ROOT / "data/processed/drug_table.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "cache/graphs/molecular_graphs.npz")
    args = parser.parse_args()
    build_graphs(args.drug_table, args.output)


if __name__ == "__main__":
    main()
