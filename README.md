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
- ChEMBL 37: official bulk molecule representations, used as a structure
  fallback when the PubChem record is absent or incomplete.
- UniChem: ChEMBL↔PubChem and DrugBank↔PubChem mapping tables and source IDs.
- SIDER 4.1: all published source tables; side-effect edges are mapped through
  the PubChem stereo identifiers.
- ClinPGx (formerly PharmGKB): chemical, gene, drug-label, and clinical
  annotation archives.
- STRING 12.5: human-only (taxon 9606) protein links, aliases, and annotations.
- UniProt: the human reference proteome FASTA.

Each raw file is left unmodified. File URLs, release dates, sizes, and SHA-256
checksums are recorded in `data/raw/DATA_PROVENANCE.yaml`. Raw and processed
datasets, generated splits, and feature/graph caches are excluded from Git; only
the raw-data checksum/provenance manifest and generation code are tracked.
Rebuild generated artifacts with the pipeline below. Follow each source's terms
before redistributing any files.

## Preprocess the downloaded data

Install the data dependencies (`pandas`, `numpy`, `requests`, and RDKit), then
run:

```powershell
python src/data/download_pubchem_properties.py
python src/data/preprocess_ddinter.py
python src/data/build_provenance.py
```

After those inputs exist, `python -m src.data.pipeline` runs the local cleaning,
checksum refresh, split generation, fingerprint generation, and molecular graph
packing stages in order. Use `--dry-run` to print the commands without running
them.

The preprocessing command writes clean drug and positive-pair tables, a full
ID/SMILES exclusion audit, STRING PPI and SIDER side-effect edge tables, and a
dataset statistics report under `data/processed/`. Generate the primary 1:1
pair splits and structure features with:

```powershell
python src/data/make_splits.py
python src/backbone/molecular_features.py
python src/graph/molecular.py
```

This creates S1/S2/S3 splits for seeds 0–9 under `data/splits/`. S2 holds out
validation and test drug cohorts from training and evaluates pairs between
each held-out cohort and drugs that occur in training pairs. S3 evaluates pairs
within disjoint held-out validation/test cohorts. Cold-start edge counts vary
with cohort assignment. Positive pairs crossing cohorts are omitted and
counted in each split's metadata. Each fold gets 1:1 unlabeled pairs
sampled without replacement, with endpoint probabilities proportional to
positive degree. Unlabeled pairs remain uncertain labels. Packed molecular
graphs are saved under `cache/graphs/`.

## Prepare a training run

The structure-only symmetric pair baseline is implemented in
`src/heads/classical.py` and `src/backbone/train_pairs.py`. For example, after
the preprocessing commands above:

```powershell
python -m src.backbone.train_pairs --split-dir data/splits/S1/seed_00 --output-dir results/ddinter/s1_seed_00
```

This command starts model training. It has not been run as part of preprocessing.
See `configs/training.yaml` for the baseline defaults.

## Current limits

The DrugBank DDI labels and drug-target/drug-enzyme tables are not present yet;
the academic account/licence is required to download them. UniChem's
DrugBank↔PubChem crosswalk is included, but it does not replace those labels.
DDInter therefore serves as the available benchmark. The pair split builder
creates sampled unlabeled examples for benchmark training, but they must not be
interpreted as confirmed non-interactions. DrugBank target/enzyme associations
and labels are absent, so the planned heterogeneous R-GCN, pretraining, and
backbone go/no-go comparison are not implemented by the structure-only baseline.
