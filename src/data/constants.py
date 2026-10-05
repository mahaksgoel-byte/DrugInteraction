"""
constants.py – Central schema definitions for raw sources and processed outputs.

Each SOURCE_SCHEMAS entry describes the expected CSV/TSV columns coming out of
a raw data file.  Downstream loaders validate against these schemas so any
upstream format change is caught early, before any data transformation happens.

CANONICAL_COLUMNS defines the output column names for the processed tables.
All cleaning / harmonisation steps write to these names.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Raw-source expected schemas
# ---------------------------------------------------------------------------

SOURCE_SCHEMAS: dict[str, list[str]] = {
    # ── DDInter ──────────────────────────────────────────────────────────────
    # Two files: code_A and code_B (different letter-range batches, same schema)
    "ddinter": [
        "DDInterID_A",
        "Drug_A",
        "DDInterID_B",
        "Drug_B",
        "Level",          # "Major" | "Moderate" | "Minor"
    ],

    # ── ChEMBL drug table (exported via chembl_client or SDF/CSV download) ──
    "chembl_drugs": [
        "chembl_id",       # e.g. CHEMBL25
        "pref_name",
        "smiles",          # canonical SMILES from ChEMBL
        "inchi_key",
        "molecule_type",   # "Small molecule" | "Protein" | …
        "max_phase",       # clinical phase (0–4)
    ],

    # ── PubChem CID mapping (from UniChem or PUG-REST batch download) ────────
    "pubchem_mapping": [
        "pubchem_cid",
        "inchi_key",
        "chembl_id",       # may be empty; populated via UniChem
        "drugbank_id",     # may be empty
        "canonical_smiles",
    ],

    # ── PharmGKB drug / gene cross-references ────────────────────────────────
    "pharmgkb_drugs": [
        "PharmGKB Accession Id",
        "Name",
        "Generic Names",
        "Trade Names",
        "Brand Mixtures",
        "Type",            # "Drug" | "Metabolite" | …
        "Cross-references", # semicolon-separated "ChEMBL:CHEMBL25" etc.
        "External Vocabulary",
        "Clinical Annotation Count",
        "Variant Annotation Count",
        "Pathway Count",
        "VIP Count",
        "Dosing Guideline",
        "Label Has Dosing Info",
        "Has Rx Annotation",
        "RxNorm Identifiers",
        "ATC Identifiers",
        "PubChem Compound Identifiers",
        "SMILES",
        "InChI",
    ],

    # ── SIDER (side-effect resource) ─────────────────────────────────────────
    # SIDER uses STITCH IDs (= PubChem CID with sign flip & padding)
    "sider_se": [
        "stitch_id_flat",   # CID without stereo
        "stitch_id_stereo", # CID with stereo
        "umls_id_label",    # UMLS concept ID from label
        "meddra_type",      # "PT" (preferred term) | "LLT"
        "umls_id_meddra",
        "side_effect_name",
    ],

    # ── STRING PPI (protein–protein interactions) ─────────────────────────────
    "string_ppi": [
        "protein1",         # Ensembl protein ID
        "protein2",
        "combined_score",   # 0–1000; keep ≥ 700 per plan
    ],

    # ── UniProt reviewed entries (Swiss-Prot dump, tab-separated) ────────────
    "uniprot_proteins": [
        "Entry",            # UniProt accession
        "Entry Name",
        "Gene Names",
        "Organism",
        "Sequence",         # AA sequence
        "Cross-reference (ChEMBL)",
        "Cross-reference (STRING)",
    ],

    # ── UniChem cross-reference mapping ──────────────────────────────────────
    "unichem_mapping": [
        "src_compound_id",  # ID in the source database
        "src_id",           # UniChem source number
        "uci",              # UniChem internal ID
    ],
}

# ---------------------------------------------------------------------------
# Canonical column names for processed/output tables
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: dict[str, list[str]] = {
    # ── drugs table (one row per unique drug) ────────────────────────────────
    "drugs": [
        "drug_id",          # our internal integer surrogate key
        "drugbank_id",      # DrugBank ID (primary, if available)
        "chembl_id",        # ChEMBL ID
        "pubchem_cid",      # PubChem CID
        "ddinter_id",       # DDInter ID (may be absent for non-DDI drugs)
        "pharmgkb_id",      # PharmGKB accession
        "name",             # preferred name (lower-cased, stripped)
        "smiles_raw",       # SMILES as downloaded (before canonicalization)
        "smiles_canon",     # RDKit-canonicalized SMILES (set after cleaning)
        "inchi_key",        # Standard InChIKey (set after cleaning)
        "mol_weight",       # computed by RDKit (Da)
        "num_atoms",        # heavy atom count
        "num_rings",        # ring count
        "is_valid_smiles",  # bool: RDKit sanitization succeeded
        "source",           # which source(s) contributed this drug
    ],

    # ── ddi_pairs table (positive DDI pairs only) ────────────────────────────
    "ddi_pairs": [
        "pair_id",          # internal surrogate key
        "drug_id_a",        # FK → drugs.drug_id  (a < b enforced)
        "drug_id_b",        # FK → drugs.drug_id
        "level",            # "Major" | "Moderate" | "Minor" | "Unknown"
        "label",            # int 1 (always 1 for positive pairs)
        "source",           # "ddinter" | "drugbank" | both
    ],

    # ── targets table (drug–protein edges) ───────────────────────────────────
    "drug_targets": [
        "drug_id",          # FK → drugs.drug_id
        "uniprot_id",       # UniProt accession
        "relation",         # "target" | "enzyme" | "carrier" | "transporter"
        "source",           # "drugbank" | "chembl" | "pharmgkb"
    ],

    # ── side_effects table ───────────────────────────────────────────────────
    "drug_side_effects": [
        "drug_id",          # FK → drugs.drug_id
        "umls_id",          # UMLS concept ID
        "side_effect_name",
        "meddra_type",      # "PT" | "LLT"
    ],

    # ── ppi table (protein–protein interaction edges) ────────────────────────
    "ppi_edges": [
        "uniprot_a",        # UniProt accession (a < b enforced)
        "uniprot_b",
        "combined_score",   # STRING combined score (0–1000)
    ],

    # ── id_mapping table (cross-reference log) ───────────────────────────────
    "id_mapping_log": [
        "step",             # e.g. "ddinter_to_chembl"
        "n_before",
        "n_after",
        "n_lost",
        "notes",
    ],
}

# ---------------------------------------------------------------------------
# Severity / interaction level ordering (ordinal)
# ---------------------------------------------------------------------------

DDI_LEVEL_ORDER: dict[str, int] = {
    "Major":    3,
    "Moderate": 2,
    "Minor":    1,
    "Unknown":  0,   # confirmed present in real DDInter data (code_A + code_B)
}

# ---------------------------------------------------------------------------
# File-name templates for processed outputs (relative to data/processed/)
# ---------------------------------------------------------------------------

PROCESSED_FILES: dict[str, str] = {
    "drugs":              "drugs.parquet",
    "ddi_pairs":          "ddi_pairs.parquet",
    "drug_targets":       "drug_targets.parquet",
    "drug_side_effects":  "drug_side_effects.parquet",
    "ppi_edges":          "ppi_edges.parquet",
    "id_mapping_log":     "id_mapping_log.csv",
    "stats_report":       "dataset_statistics.json",
}
