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
counted in each split's metadata. Each fold gets 1:1 unlabeled pairs sampled
uniformly without replacement from legal cohort pairs. This avoids using
held-out positive degrees to decide which test negatives to include. Unlabeled
pairs remain uncertain labels. Packed molecular
graphs are saved under `cache/graphs/`.

## Train the feature-based baseline

The symmetric pair-feature MLP is implemented in
`src/heads/classical.py` and `src/backbone/train_pairs.py`. Feature generation
now writes separate ablations and a combined input: Morgan fingerprints (2,048
bits), SIDER adverse-effect terms (up to 1,024 plus a source-availability flag),
and ClinPGx/PharmGKB clinical gene annotations (up to 512 common genes, six
clinical categories, plus an availability flag). ClinPGx drug names are joined
to canonical drugs by exact normalized aliases only; ambiguous aliases and
combination names are skipped. In the current data this yields annotations for
337 of 1,487 drugs, so it is a sparse source. It can add pharmacogenomic context,
but is not a drug-target database. The combined feature file has 3,592 columns;
Morgan + SIDER without ClinPGx is also saved for a controlled ablation. These
are single-drug descriptors, not pair labels. For example, after preprocessing:

```powershell
python -m src.backbone.train_pairs --split-dir data/splits/S1/seed_00 --output-dir results/ddinter/s1_seed_00
```

The trainer logs every epoch to the terminal and incrementally to
`epoch_metrics.jsonl`, selects the checkpoint with the best validation AUPRC,
stops when validation has not improved for the configured patience, then evaluates
that checkpoint on the held-out test set. It writes `best.pt` and
`training_history.json` (including per-epoch train loss, validation metrics, and
test metrics) under the output directory. The classification threshold is
selected on validation and fixed before test evaluation. The test split is used
only after checkpoint selection.

### Training dashboard

Install Streamlit if needed (`pip install streamlit`), then run from the project
root:

```powershell
streamlit run training_app.py
```

The dashboard has Preprocessing, Training, Classical vs Quantum, and Project roadmap sections in its sidebar. Preprocessing
reports which local artifacts are present and can run the pipeline against files
already downloaded to `data/raw`. Training lets you choose S1/S2/S3, split seed,
feature set, model width, learning rate, dropout, and weight decay. It ranks
configurations on validation AUPRC, then evaluates the selected configuration
once on the held-out test split. The complete epoch archive lists every search
trial and final fit, with per-epoch train/validation loss, ranking and classification
metrics, threshold, learning rate, and elapsed time. The page can compare saved
test runs and explains why cohort, setting, feature and training differences can
change a score. It explains every epoch-history column and training control in
plain language. New runs also save Matthews correlation, Brier score and
10-bin expected calibration error; old runs do not contain these values. JSON
and JSONL logs remain under `results/ddinter/` after the UI closes. The roadmap
page marks implemented, partial and blocked parts of `plan.md`. The current model
is a classical feature-based pair model. The Qiskit section is a separate
experimental four-qubit circuit comparison on compact molecular pair inputs;
the planned frozen heterogeneous graph and full research design are not built.

The CLI exposes the same training settings (`--epochs`, `--patience`,
`--batch-size`, `--learning-rate`, `--width`, `--dropout`, and `--weight-decay`).
The benchmark uses sampled unlabeled pairs as negatives, so test scores describe
this sampled DDInter task and are not clinical safety probabilities.

For a headless validation-only width/learning-rate grid search, use
`python -m src.backbone.search_pairs --split-dir data/splits/S3/seed_00
--output-dir results/ddinter/s3_search`. The script ranks configurations by
validation AUPRC and does not evaluate the test split during model selection.

The Training page also runs a degree-only shortcut diagnostic. It computes each
drug's number of positive pairs in the training partition only, learns a small
logistic model from those counts, chooses its threshold on validation, and
reports the held-out set for diagnosis. On seed 2 it scores 79.64% in S1,
70.01% in S2, and 50.00% in S3. No S3 test drug has a positive training degree,
so chance performance is expected. This helps rule out *this particular*
popularity shortcut in S3; it does not replace automated leakage tests.

The split audit is implemented in `src/data/audit_splits.py` and the saved
`results/ddinter/split_integrity_audit.json` records 30 / 30 splits passing. It
checks source checksums, canonical and non-repeated pairs, known positives not
being used as unlabeled examples, balanced 1:1 sampling, and S2/S3 drug-cohort
rules. This validates the generated split files against the current local data;
it is not a proof that DDInter's underlying labels are complete.

Re-run all default split audits from the project root with:

```powershell
python -m src.data.audit_splits --output results/ddinter/split_integrity_audit.json
```

Run an individual degree-only baseline with:

```powershell
python -m src.backbone.degree_baseline --split-dir data/splits/S3/seed_02 --output results/ddinter/degree_baselines/s3_seed_02.json
```

### Results and their limits

The following baseline runs use a 1:1 sample of unlabeled pairs, Morgan + SIDER
features (before ClinPGx was added), and thresholds selected on validation data.
S3 seed 1 is repeated on a separate cohort split; none of these constitute a
completed ten-seed study:

| Setting | Seed | Best epoch / completed | Validation accuracy | Test accuracy | Test AUROC | Test AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| S1 warm-start | 0 | 13 / 21 | 92.95% | 92.87% | 0.9782 | 0.9752 |
| S2 one-drug cold-start | 0 | 11 / 23 | 74.97% | 76.75% | 0.8371 | 0.8453 |
| S3 strict cold-start | 0 | 7 / 15 | 65.32% | 64.77% | 0.6964 | 0.7102 |
| S3 strict cold-start | 1 | 7 / 19 | 61.13% | 64.85% | 0.6991 | 0.7083 |

S1 allows the same drugs to appear in training and test pairs, so its high score
does not establish performance on new drugs. S2 tests a pair containing one
held-out drug and one training-seen drug. S3 holds out both drugs in each test
pair and is the relevant result for a strict cold-start claim. In S3,
the Morgan-only ablation scored 59.75% test accuracy / 0.6385 AUPRC; SIDER-only
scored 60.91% / 0.6665; the seed-0 combined baseline scored 64.77% / 0.7102.
Those three ablations each use one seed, so repeated-seed confidence intervals
and external validation remain necessary before making a broad performance
claim. The two original Morgan + SIDER S3
seed test accuracies average 64.82%; two seeds are still too few for a stable
estimate, but they show that the low strict cold-start score repeats across a
different drug cohort assignment.

The S2 run used the same fixed Morgan + SIDER model configuration (width 128,
learning rate 0.001, batch size 512, dropout 0.2, weight decay 0.01) and stopped
after 23 epochs at the 12-epoch patience limit. Validation AUPRC peaked at epoch
11; later training loss fell while validation performance did not recover. Its
held-out test accuracy is 76.75%, still below the requested range.

I also tested encoder-level dropout on S3 seed 1 to reduce memorization inside
the shared drug encoder. It selected epoch 8 of 20; test accuracy was 65.03%
and AUPRC 0.7072 versus 64.85% and 0.7083 for the prior architecture. This is
not a meaningful accuracy gain, though validation loss was less extreme. The
change is retained as a regularization option, not presented as a breakthrough.

Adding the exact-name-mapped ClinPGx gene/category features to that regularized
model produced a stronger S3 seed-1 result: validation selected epoch 26 of 38,
with 62.85% validation accuracy; held-out test accuracy was 66.40%, AUROC
0.7062, and AUPRC 0.7318. Against the same-seed Morgan + SIDER regularized run,
test accuracy improved by 1.55 percentage points and AUPRC by 0.0246. This is
encouraging but remains one seed and far below 80-85%; it does not establish
robust benefit from PharmGKB.

The S2 seed-0 run with the same updated features/model selected epoch 21 of 33.
Its held-out accuracy was 76.82% and AUPRC 0.8456, essentially unchanged from
the Morgan + SIDER run (76.75% / 0.8453). Thus ClinPGx helped this one S3 split
but did not materially improve S2; it should stay an ablation until repeated
seeds confirm its value.

### Fresh S3 regularization experiment (seed 2)

To check whether a smaller model and stronger dropout could close the S3 gap,
I ran a four-configuration search using only the seed-2 training and validation
files. All trials used Morgan + SIDER + ClinPGx, batch size 1,024, learning
rate 0.001, and weight decay 0.01; widths were 64 or 128 and dropout was 0.20
or 0.35. Trials were ranked by validation AUPRC. No test result was read until
the configuration had been selected. The complete epoch-by-epoch logs are in
`results/ddinter/s3_seed_02_regularization_validation/`; the final selected run
is in `results/ddinter/s3_seed_02_regularization_final/`.

The selected setup was width 64 and dropout 0.35. It selected epoch 1 of 9:
validation accuracy was 68.49%, validation AUROC 0.7470, and validation AUPRC
0.7572. On the untouched seed-2 test cohort it scored 63.22% accuracy, AUROC
0.6668, and AUPRC 0.6678. The validation-to-test drop is a warning that one
validation cohort does not reliably predict another held-out drug cohort.
This experiment did not reach 70%, and the model should not be described as
having done so. The best saved S3 test accuracy remains 66.40% on seed 1.
The seed-2 outcome is lower, reinforcing that S3 performance varies by which
drugs are held out. Do not keep tuning against these test scores; use additional
validation cohorts, then reserve a new untouched cohort for a final claim.

Why validation was higher: the seed-2 validation set scored 68.49% accuracy,
while its separate test set scored 63.22%, a 5.27 percentage-point gap. The
validation and test sets contain different held-out drugs and different sampled
unlisted pairs, so one cohort can simply be easier for the available features.
The four-configuration search also selected the best setting on validation,
which can make that validation score somewhat optimistic. The threshold was
chosen on validation and then frozen. Seed 1's 66.40% test score used a different
drug cohort and model setting, so its 3.18-point difference from seed 2 cannot
be assigned to one cause. Repeated same-protocol cohorts are needed to estimate
the typical score and separate model changes from cohort difficulty.

This experiment also illustrates why epochs are not a performance target. The
selected checkpoint was from epoch 1 even though training continued to epoch 9
under early stopping. Later epochs made training loss fall but did not improve
validation AUPRC. Running more epochs would have kept fitting the training pairs
without evidence of better generalization.

The desired 80-85% accuracy should not be forced by training for more epochs.
The initial S3 baseline reached its best validation checkpoint at epoch 7 and
stopped at epoch 15 (seed 0) and epoch 19 (seed 1); both test accuracies remained
about 65%. S1 is above 85% because its split
allows drug identities to recur across train and test. A higher S3 score requires
new predictive information and repeated held-out evaluation, not choosing epochs
or thresholds against the test set. Accuracy also depends on this benchmark's
artificial 1:1 unlabeled-pair sampling.

The selected inputs are pivotal for this task because they exist per drug before
the pair label is known: chemical substructures (Morgan) and independent
single-drug adverse-effect terms (SIDER). In the seed-0 S3 ablation, Morgan +
SIDER reached 64.77% accuracy / 0.7102 AUPRC; Morgan alone reached 59.75% /
0.6385, and SIDER alone 60.91% / 0.6665. This supports retaining both sources
for the next iteration, but one seed is not evidence of robust superiority.
DDInter severity is an outcome label and must not be used as an input feature.
The available STRING data are protein-protein links, not drug-target links, so
they cannot directly provide drug mechanism features. Drug-target/enzyme edges
and a separate DrugBank DDI benchmark are currently unavailable under the local
data access/licensing setup. The most useful next modeling route is to audit and
add mapped drug-target, transporter, and enzyme evidence, then evaluate a drug
pair model using target/pathway and molecular features under the same S3 split;
in parallel, use a positive-unlabeled objective or carefully justified
confidence weights because unlisted DDInter pairs are not verified negatives.
Report multiple split seeds with mean and confidence intervals and, ideally,
an external temporal or source-held-out benchmark before claiming state of the art.

## Current limits

The Training UI also includes the newer **Cold-start accuracy** experiment:
regularized boosted trees on training-selected chemical/side-information
features, training-only similarity transfer, and validation-selected model
averages. Its complete logs and reports are in
`results/cold_start_improvement/validation_trees_transfer_v1/`. The algorithm
selection, feature coverage, per-cohort results and train/validation/test gaps
are explained in the UI. This uses all training pairs; do not present it as a
matched-budget quantum comparison. Reproduce it with the two commands in the
accuracy improvement extension of `plan.md`.

The completed validation-selected tree/similarity blend scores **65.41%, 66.54%,
65.99%** on S3 test seeds 0/1/2 (mean **65.98%**). Validation scores are 70.78%,
67.09%, 71.50%. The gap matters: these experiments have not achieved 70–85%
test accuracy. All base trials, ensemble choices, checkpoints, predictions and
feature sensitivity records are preserved. The current shared neural checkpoints
score 61.98%, 66.40%, 63.22% on these cohorts; their separate settings and the
historical reuse of these cohorts make this an exploratory comparison.

The DrugBank DDI labels and drug-target/drug-enzyme tables are not present yet;
the academic account/licence is required to download them. UniChem's
DrugBank↔PubChem crosswalk is included, but it does not replace those labels.
DDInter therefore serves as the available benchmark. The pair split builder
creates sampled unlabeled examples for benchmark training, but they must not be
interpreted as confirmed non-interactions. DrugBank target/enzyme associations
and labels are absent, so the planned heterogeneous R-GCN, pretraining, and
backbone go/no-go comparison are not implemented by the structure-only baseline.

## Classical vs quantum experiment

### Current eight-qubit comparison and demonstration

Open **Classical vs Quantum** in the sidebar. The current comparison freezes a
trained drug feature encoder, compresses its summaries using training-only PCA,
and fits classical RBF and real Qiskit fidelity-kernel SVM heads on the same
1,024 training pairs. Both receive nine validation trials. The complete saved
validation and test folds are evaluated. Logistic regression and removal of
cross-register entanglers provide additional controls.

The distinction is explicit: classical RBF uses ordinary mathematical
similarity; quantum fidelity uses state overlaps from an eight-qubit Qiskit
circuit with two drug registers. Both SVM optimizers run classically. The
circuit is simulated locally; no quantum processor, speedup or general quantum
advantage is claimed. This is a frozen feature MLP, not the planned GNN.

The UI includes cohort comparisons, a replay of every saved held-out prediction,
the circuit diagram, complete validation trials, the shared encoder history,
report downloads, noise sensitivity, classical imitation and confidence
calibration. SVM solver iterations are not epochs. All earlier neural and
variational epochs remain available in Training and the pilot archive.

```powershell
cd C:\MehakDS\DrugInteraction
.\.quantum-venv\Scripts\python.exe -m streamlit run training_app.py
```

To watch a new comparison in your terminal, choose an unused output directory:

```powershell
.\.quantum-venv\Scripts\python.exe -m src.backbone.quantum_kernel --checkpoint-dir results/ddinter/s3_seed_02_regularization_final --output-dir results/quantum_kernels/my_new_run --cap 1024
.\.quantum-venv\Scripts\python.exe -m src.analysis.quantum_diagnostics results/quantum_kernels/my_new_run
.\.quantum-venv\Scripts\python.exe -m unittest tests.test_dashboard tests.test_quantum_kernel -v
```

Each run saves `protocol.json`, `events.jsonl`, `validation_trials.json`,
`models.joblib`, pair samples, full `test_predictions.csv.gz`, and
`comparison.json`. The diagnostics command adds `diagnostics.json` and its
fixed validation subset. Checkpoint, feature and split hashes identify inputs;
existing run directories are not overwritten. Only load trusted local joblib
and PyTorch files. Absolute artifact paths currently assume this workspace;
moving the project requires updating paths or rerunning the comparison.

Noise studies use a fixed 512-pair validation subset: 100/1,000/10,000 overlap
shots with exact normalization, and a simplified global query-state
depolarization channel. They do not model a real device. Classical imitation
is tested on that subset; calibration is fitted there and evaluated on full
test. Cohorts overlap and encoders were selected separately, so cohort means
are descriptive, not a significance test. The original plan's outstanding
research requirements are explicitly recorded in `plan.md`.

### Completed eight-qubit results

The newer completed eight-qubit experiments report these full-test accuracies:

| S3 cohort | Classical RBF | Quantum fidelity | Classical logistic |
|---|---:|---:|---:|
| Seed 0 | 61.07% | 61.70% | 61.72% |
| Seed 1 | 66.13% | 66.96% | 65.29% |
| Seed 2 | 62.92% | 62.88% | 63.42% |

Quantum has a small descriptive mean gain over RBF, not consistent superiority
over all classical models. The 70–85% S3 target remains unmet. Cross-register
entanglement does not consistently improve scores. Detailed limitations and
the remaining original research scope are in the UI and `plan.md`.

### Earlier four-qubit pilot

The archived pilot in the **Classical vs Quantum** sidebar compares logistic regression
with a four-qubit variational circuit executed by Qiskit's exact Statevector
simulator. This is genuine simulation of a Qiskit quantum circuit on ordinary
hardware; it is not a quantum processor. Both models use the same selected
S1/S2/S3 split, the same capped stratified pair samples, and validation-selected
thresholds. PCA and feature scaling are fitted on training data only. Epoch
history and fitted parameters are saved under `results/ddinter/qiskit_*`.

Because the shared Python environment's NumPy and PyTorch binaries are not
compatible with recent Qiskit Machine Learning releases, use the isolated
project environment and the pinned Qiskit core dependency:

```powershell
python -m venv --system-site-packages .quantum-venv
.\.quantum-venv\Scripts\python.exe -m pip install -r requirements-quantum.txt
.\.quantum-venv\Scripts\python.exe -m streamlit run training_app.py
```

The UI identifies the exact number of sampled train, validation, and test pairs.
These capped results are exploratory; repeat across several seeds and then use
an untouched larger evaluation before making performance claims. No quantum
hardware access is configured.
