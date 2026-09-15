"""Stage 3 — Molecular parsing with RDKit.

Parses SMILES into atom-bond graphs and computes Morgan fingerprints as
the fallback/aux drug embedding (ChemBERTa is the production choice per
the spec; Morgan is used here to keep the demo dependency-light and fast).
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

# The synthetic fallback dataset (data_sources.py) fabricates SMILES-like
# strings for demo purposes; RDKit correctly flags many as chemically
# invalid (bad valence). Real source data won't trigger this. Silenced so
# it doesn't drown out the pipeline's own stage logs.
RDLogger.DisableLog("rdApp.*")

ATOM_VOCAB = ["C", "N", "O", "F", "S", "Cl", "Br", "P", "I", "B", "UNK"]
BOND_VOCAB = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def parse_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    return mol


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """One-hot element + charge + hybridization, per Stage 3 spec."""
    symbol = atom.GetSymbol()
    one_hot = [1.0 if symbol == v else 0.0 for v in ATOM_VOCAB[:-1]]
    one_hot.append(1.0 if symbol not in ATOM_VOCAB[:-1] else 0.0)  # UNK bucket
    charge = [float(atom.GetFormalCharge())]
    hybridization = [float(int(atom.GetHybridization()))]
    aromatic = [1.0 if atom.GetIsAromatic() else 0.0]
    return np.array(one_hot + charge + hybridization + aromatic, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    one_hot = [1.0 if bond.GetBondType() == v else 0.0 for v in BOND_VOCAB]
    return np.array(one_hot, dtype=np.float32)


def mol_to_graph(mol: Chem.Mol) -> dict:
    """Atom-bond graph: nodes with element/charge/hybridization features,
    edges with bond-type features (undirected, both directions added)."""
    node_feats = np.stack([atom_features(a) for a in mol.GetAtoms()])
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        edge_index += [(i, j), (j, i)]
        edge_attr += [f, f]
    edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(edge_attr) if edge_attr else np.zeros((0, len(BOND_VOCAB)), dtype=np.float32)
    return {"node_feats": node_feats, "edge_index": edge_index, "edge_attr": edge_attr}


def morgan_fingerprint(mol: Chem.Mol, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_fingerprint(smiles: str, n_bits: int = 1024) -> np.ndarray | None:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return morgan_fingerprint(mol, n_bits=n_bits)
