# DDI Prediction Pipeline — Data to Classical vs. Quantum Comparison

Implementation of the architecture in `../DDI_Pipeline_Architecture.html`
and `../DDI Prediction Pipeline Architecture.png`: a heterogeneous GNN
backbone (Stages 1-9) whose 8-dim bottleneck embedding feeds two parallel
drug-drug-interaction (DDI) prediction heads — a classical MLP and a
Qiskit variational quantum circuit — evaluated side by side.

## Layout

```
ddi_pipeline/
  configs/default.yaml       # every stage's hyperparameters
  data/raw/                  # put DrugBank/ChEMBL/PharmGKB/SIDER exports here (optional)
  data/processed/
  outputs/                   # comparison_results.csv lands here
  scripts/run_pipeline.py    # end-to-end runner
  src/ddi_pipeline/
    config.py                # config + seeding
    data_sources.py          # Stage 1: loaders + synthetic fallback
    preprocessing.py         # Stage 2 + cleaning/balance/split
    molecular.py             # Stage 3: RDKit parsing + Morgan fingerprints
    graph_construction.py    # Stage 4-5: HeteroData graph + node features
    gnn_backbone.py          # Stage 6-9: pretrain, R-GCN encoder, pooling, bottleneck
    classical_head.py        # Pipeline A
    quantum_head.py          # Pipeline B (Qiskit EstimatorQNN + TorchConnector)
    training.py               # shared training loop
    evaluation.py              # Stage 10: comparison table
```

## Setup

```bash
cd ddi_pipeline
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

Real DrugBank/PharmGKB downloads require a (free) account and manual
export — if `data/raw/` doesn't contain the expected CSVs (see docstrings
in `data_sources.py` for exact filenames/columns), the pipeline
auto-generates a small structurally-identical **synthetic** dataset so
the full pipeline runs end to end for the demo. Swap in real exports by
dropping the CSVs into `data/raw/` — no code changes needed.

Output: `outputs/comparison_results.csv` — accuracy, F1, AUROC, AUPRC,
cold-start metrics, parameter count, and training time for both heads.

## Design notes

- **Fair comparison**: the shared GNN backbone is pretrained
  (self-supervised, Stage 6) then fine-tuned once using the classical
  head as a supervised probe, then **frozen**. Both heads then train on
  the identical frozen 8-dim bottleneck tensor — this is what the
  diagram's "Fork point: identical 8-dim input to both heads" note is
  ensuring.
- **Quantum head** runs on Qiskit's Aer simulator only — there is no real
  quantum hardware call and no expected speed benefit; the research
  question is accuracy under an identical backbone/data/split, not runtime.
- **Cold-start split**: a fraction of drugs are withheld from training
  entirely; any pair touching a held-out drug goes to `cold_start_test`,
  which is the harder and more meaningful generalization test (per
  HetDDI's S1/S2/S3 design referenced in the architecture doc).
