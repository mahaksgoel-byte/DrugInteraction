# Drug Interaction: classical vs quantum heads

The project protocol is in [plan.md](plan.md). This checkout currently contains
the first data-collection and cleaning pass for the public DDInter benchmark.

## Downloaded inputs

- DDInter 2.0: all eight official interaction category CSV files.
- DDInter drug metadata crosswalk: supplemental scraper export with DDInter,
  DrugBank, ChEMBL, and PubChem identifiers; its third-party provenance is
  called out in `data/raw/DATA_PROVENANCE.yaml`.
- PubChem PUG REST: batched title, structure, and InChIKey responses for the
  DDInter crosswalk compounds.
- UniChem: ChEMBL↔PubChem and DrugBank↔PubChem mapping tables and source IDs.
- SIDER 4.1: all published source tables; side-effect edges are mapped through
  the PubChem stereo identifiers.
- ClinPGx (formerly PharmGKB): chemical, gene, drug-label, and clinical
  annotation archives.
- STRING 12.5: human-only (taxon 9606) protein links, aliases, and annotations.
- UniProt: the human reference proteome FASTA.

Each raw file is left unmodified. File URLs, release dates, sizes, and SHA-256
checksums are recorded in `data/raw/DATA_PROVENANCE.yaml`. Raw and processed
data are excluded from Git by `.gitignore`; follow each source's terms before
redistributing any files.

## Preprocess the downloaded data

Install the data dependencies (`pandas`, `numpy`, `requests`, and RDKit), then
run:

```powershell
python src/data/download_pubchem_properties.py
python src/data/preprocess_ddinter.py
python src/data/build_provenance.py
```

The preprocessing command writes clean drug and positive-pair tables, a full
ID/SMILES exclusion audit, STRING PPI and SIDER side-effect edge tables, and a
dataset statistics report under `data/processed/`.

## Current limits

The DrugBank DDI labels and drug-target/drug-enzyme tables are not present yet;
the academic account/licence is required to download them. UniChem's
DrugBank↔PubChem crosswalk is included, but it does not replace those labels.
DDInter therefore serves as the available benchmark for this pass. No negative
pairs or train/validation/test splits have been generated, and unlisted pairs
must not be treated as confirmed negatives.
