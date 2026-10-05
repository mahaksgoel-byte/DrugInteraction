# Substrate or Structure? Classical vs Quantum Heads for Drug-Drug Interaction Prediction
### Final end-to-end project plan (data collection → final results)

---

## 0. Project summary

**Core question:** Once a strong heterogeneous GNN has produced drug embeddings, does a *quantum* head add anything over a *classical* head, and is any difference due to the quantum **substrate** or to the **pair-structured inductive bias** of the head?

### Novelty points (all must be present in the final project)

| # | Novelty element | Where it lives |
|---|---|---|
| N1 | **Fair fork point:** one frozen GNN backbone, identical cached embeddings fed to every head | Step 5 |
| N2 | **Pair-native, swap-symmetric quantum ansatz:** two 4-qubit registers (Drug A / Drug B), shared weights, cross-register entanglement as the "interaction", f(A,B) = f(B,A) | Step 6 |
| N3 | **2×2 factorial design:** {classical, quantum} × {generic head, pair-structured head} to separate substrate from inductive bias | Step 6 |
| N4 | **Kernel arms:** RBF-SVM vs fidelity quantum-kernel SVM on the same features | Step 6 |
| N5 | **Parameter-matched capacity sweeps** (performance vs parameter count, not a single matched point) | Step 7 |
| N6 | **Strict cold-start evaluation** (unseen drugs) with leakage-free graphs and degree-matched negatives | Steps 3–4 |
| N7 | **Data-efficiency curves** (1% → 100% training data) testing the "quantum helps in low data" claim | Step 8 |
| N8 | **Qubit-scaling study** (8 / 10 / 12 qubits vs matched classical) | Step 8 |
| N9 | **Entanglement ablation + entanglement entropy** across the A\|B cut | Step 9 |
| N10 | **Classical simulability / surrogate test** | Step 9 |
| N11 | **Trainability analysis** (gradient variance, effective dimension) | Step 9 |
| N12 | **Noise robustness** (shot + device noise) and optional hardware inference | Step 9 |
| N13 | **Calibration (ECE)** reported alongside accuracy metrics | Step 8 |
| N14 | **Equivalence testing (TOST)** so ties are statistically supported, not just "not significant" | Step 10 |

### Hypotheses (frozen before running experiments)

- **H1:** Pair-structured heads outperform generic heads in both substrates.
- **H2:** At matched parameters, quantum and classical heads differ by less than a pre-set margin δ (equivalence).
- **H3:** Any quantum advantage appears only in low-data or strict cold-start regimes.

---

## 1. Environment and repository setup

**Goal:** reproducible infrastructure before any data is touched.

1. Create a Git repo with this layout:
   ```
   ddi-quantum/
   ├── configs/            # Hydra YAML configs (data, backbone, heads, experiments)
   ├── data/
   │   ├── raw/            # untouched downloads + checksums
   │   ├── processed/
   │   └── splits/         # saved split indices (S1/S2/S3 × seeds)
   ├── src/
   │   ├── data/           # download, harmonize, clean, negatives, splits
   │   ├── graph/          # RDKit parsing, hetero graph build, features
   │   ├── backbone/       # pretraining, R-GCN encoder, embedding export
   │   ├── heads/          # classical + quantum heads, kernels
   │   ├── analysis/       # entanglement, surrogate, trainability, noise
   │   └── stats/          # CIs, Wilcoxon, Holm, TOST
   ├── cache/embeddings/   # frozen per-drug embeddings
   ├── results/            # raw metrics (one row per run)
   ├── notebooks/
   └── tests/              # symmetry, leakage, shape tests
   ```
2. Environment: Python 3.10+, PyTorch, PyTorch Geometric, RDKit, PennyLane (+ `default.qubit`), scikit-learn, Optuna, Hydra, W&B or MLflow, DVC.
3. Fix global seeds (Python, NumPy, PyTorch); log library versions.
4. Write the **frozen experiment protocol** (one page, committed to Git *before* running experiments):
   - Hypotheses H1–H3
   - Primary metrics: AUPRC and AUROC on strict cold-start (S3)
   - Equivalence margin δ (default: 0.01 AUROC)
   - Seeds: 10; tuning budget: 50 Optuna trials per arm
   - Statistical tests (Step 10)

**Deliverable:** working repo, environment lockfile, committed protocol.

---

## 2. Data collection and cleaning

### 2.1 Sources

| Source | Purpose | Notes |
|---|---|---|
| DrugBank | DDI labels, drug-target, drug-enzyme | Academic licence, account required |
| DDInter | Second DDI dataset (independent benchmark) | Free |
| ChEMBL / PubChem | SMILES, ID mapping | ChEMBL client, PubChem PUG-REST, UniChem |
| PharmGKB | CYP enzyme status | Academic use |
| SIDER | Single-drug side-effect profiles | Note age of data |
| STRING | PPI edges | Confidence ≥ 700 |
| UniProt | Protein sequences | For protein embeddings |

Optional third benchmark: a standard DDI benchmark (e.g. via TDC or BioSNAP). **Never use side-effect features derived from the evaluation dataset itself** (label leakage).

### 2.2 Steps

1. Download everything into `data/raw/`; record source URL, version/date and SHA-256 checksum.
2. **ID harmonization:** unify DrugBank ↔ PubChem ↔ ChEMBL IDs. Log how many drugs are lost at each mapping.
3. **Cleaning:**
   - Drop drugs without valid SMILES (RDKit sanitization must succeed).
   - Canonicalize SMILES; deduplicate drugs.
   - Canonicalize DDI pair order (sorted IDs), deduplicate pairs, drop self-pairs.
4. **Dataset statistics report:** number of drugs, positive pairs, degree distribution, targets/enzymes per drug. Save as a table.

**Deliverable:** `data/processed/` with clean drug table, DDI positive table, KG edge tables, plus a stats summary.

---

## 3. Splits and negative sampling (leakage-free)

### 3.1 Evaluation settings

| Setting | Definition | Role |
|---|---|---|
| **S1 (warm)** | Random pair split | Sanity check |
| **S2 (semi-cold)** | One drug of each test pair unseen in training | Secondary |
| **S3 (strict cold)** | Both drugs unseen in training | **Primary** |
| Optional | Scaffold-based drug split | Robustness |

Generate **10 split seeds per setting**, 70/10/20 (train/val/test), and save indices to `data/splits/`.

### 3.2 Negatives

1. DrugBank provides only positives, so sample negatives from unlisted pairs.
2. **Primary:** 1:1 ratio, **degree-matched** (negatives drawn so each drug's positive/negative count is balanced, preventing popularity shortcuts).
3. **Sensitivity:** 1:5 ratio.
4. Acknowledge false negatives (an unlisted pair is not proven non-interacting).
5. **Shortcut check:** train a degree-only model. If it scores well above chance, redo negative sampling.

### 3.3 Graph hygiene rules (enforced by tests)

- In S2/S3, **remove every DDI edge touching held-out drugs** from the message-passing graph, from pretraining, and from any feature computation.
- Held-out drugs keep structure, target, enzyme and side-effect information (standard cold-start assumption).
- Run a **structure-only sensitivity variant** (held-out drugs keep only molecular graphs).
- Write unit tests: no held-out drug appears in training pairs; no test DDI edge exists in the training graph.

**Deliverable:** saved splits, negative sets, passing leakage tests, degree-only baseline report.

---

## 4. Molecular and heterogeneous graph construction

1. **Molecular graphs:** RDKit SMILES → atom-bond graph (atom type, degree, charge, aromaticity, hybridization; bond type, ring, conjugation).
2. **Heterogeneous graph (PyTorch Geometric `HeteroData`):**
   - Nodes: Drug, Protein, Enzyme, SideEffect
   - Edges: `targets`, `metabolized_by`, `causes`, `ppi`
   - `interacts_with` is the **prediction target** only (never a message-passing edge for held-out pairs)
3. **Protein/enzyme features:** ESM-2 (small variant) or ProtBERT embeddings, mean-pooled over residues, cached to disk.
4. **Drug feature vector:** molecular-encoder embedding + CYP flags + side-effect multi-hot vector.
5. Build one graph object per (setting, seed) so each respects its own hygiene rules.

**Deliverable:** cached hetero-graphs and feature tensors with shape/NaN checks.

---

## 5. Backbone: self-supervised pretraining, encoder, freezing (N1)

1. **Self-supervised pretraining** (non-DDI objectives only, training drugs only):
   - Masked atom/bond feature reconstruction on molecular graphs
   - Masked edge prediction on the knowledge graph
2. **Encoder:** relation-aware R-GCN, 3 layers.
3. **Per-drug bottleneck:** output a **4-dim embedding per drug** (z_A, z_B), *not* one pooled pair vector. This is required for the pair-structured heads.
   - Generic heads receive `[z_A, z_B]` (8-dim) with random A/B order swapping during training.
4. **Backbone training:** train with a neutral probe (logistic or simple bilinear) on the training pairs only.
5. **Freeze and cache** all drug embeddings to `cache/embeddings/` per (setting, seed). **Every head trains from scratch on identical cached features.** This is the fair fork point.
6. **Go/no-go gate:** backbone + simple head must be competitive with established DDI baselines (e.g. SSI-DDI, SumGNN, Decagon-style models) on the same splits. If not, fix the backbone (simplify, drop pretraining, or use a pretrained molecular encoder) *before* building quantum components.
7. **Secondary experiment (confound check):** also fine-tune each head end-to-end through the backbone and report both frozen and end-to-end results.
8. **Scaling variants:** additionally train 5-dim and 6-dim per-drug backbones for the 10- and 12-qubit studies (Step 8).

**Deliverable:** frozen embeddings, backbone benchmark table, gate decision recorded.

---

## 6. Heads: the 2×2 design plus kernel arms (N2, N3, N4)

### 6.1 Arm definitions

| Arm | Substrate | Structure | Description |
|---|---|---|---|
| **C-G** | Classical | Generic | MLP on `[z_A, z_B]`, e.g. 8→16→1 with ReLU and dropout |
| **C-S** | Classical | Pair-structured | Shared φ on each drug; symmetric combination `[φ(a)+φ(b), φ(a)⊙φ(b), \|φ(a)−φ(b)\|]` → linear |
| **Q-G** | Quantum | Generic | Angle encoding of 8-dim vector, EfficientSU2 ansatz, Pauli-Z readout, linear post-processing |
| **Q-S** | Quantum | Pair-structured | Two 4-qubit registers, shared weights, cross-register entanglement, swap-symmetric observable |
| **C-K** | Classical | Kernel | RBF-kernel SVM |
| **Q-K** | Quantum | Kernel | Fidelity quantum kernel + SVM, swap-symmetrized feature map |
| Floors | – | – | Logistic regression; degree-only model |
| Ablations | – | – | Q-S with cross-register entanglers removed; random-feature model |

### 6.2 Q-S circuit (reference implementation sketch, PennyLane)

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=8)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def q_head(za, zb, w):                       # w shape: (L, 4, 4)
    # Encode each drug on its own register
    for i in range(4):
        qml.RY(za[i], wires=i)
        qml.RY(zb[i], wires=i + 4)
    for l in range(w.shape[0]):
        for i in range(4):
            for off in (0, 4):               # SAME weights on both registers
                qml.RX(w[l, i, 0], wires=i + off)
                qml.RY(w[l, i, 1], wires=i + off)
                qml.IsingZZ(w[l, i, 2], wires=[i + off, (i + 1) % 4 + off])
            qml.IsingXX(w[l, i, 3], wires=[i, i + 4])   # cross-register "interaction"
    obs = sum(qml.PauliZ(i) @ qml.PauliZ(i + 4) for i in range(4)) \
        + sum(qml.PauliZ(i) for i in range(8))
    return qml.expval(obs)
```

Then apply a learnable scale and bias and a sigmoid. With L=3 this is ~48 circuit parameters.

**Mandatory tests:**
- `f(a, b) == f(b, a)` to 1e-6 for random inputs.
- Output range and gradient-flow checks.

**Speed:** implement a batched PyTorch statevector simulator (256 amplitudes) and validate against PennyLane on a small batch. Looping single QNodes over tens of thousands of pairs is too slow.

### 6.3 Kernel arms

- Kernel matrices are N×N, so train kernel arms on **subsamples (5–10k pairs)** and also compare *all* arms at that same training size.
- Q-K feature map: the same pair-structured encoding as Q-S (symmetrized), kernel = state fidelity.

### 6.4 Tuning (identical budgets)

- Every arm gets **50 Optuna trials** on the validation set.
- Classical: width, learning rate, weight decay, dropout.
- Quantum: depth L, learning rate, input scaling, initialization scale (near-identity init).
- Kernel: C, gamma / feature-map scaling.

**Deliverable:** all arms implemented, symmetry/unit tests passing, tuned configs saved.

---

## 7. Parameter-matched capacity sweeps (N5)

1. For each arm, sweep capacity so that trainable parameter counts span roughly **20 → 200**:
   - Classical: vary hidden width / φ width.
   - Quantum: vary depth L (and, for Q-G, ansatz repetitions).
2. Do **not** match at a single point. Report performance-vs-parameters curves for all arms.
3. Report exact trainable parameter counts for every configuration (including post-processing layers).
4. Include the classical logistic-regression floor and the degree-only model on every plot.

**Deliverable:** parameter-count table and capacity-sweep results.

---

## 8. Main experiments (N6, N7, N8, N13)

### 8.1 Full grid

Arms (Step 6) × settings (S1, S2, S3) × datasets (DrugBank, DDInter) × **10 seeds**, at the tuned configurations and across the capacity sweep.

### 8.2 Data-efficiency study (N7)

Train on **1%, 5%, 10%, 25%, 100%** of the training pairs for every arm, under S3 and S2. This directly tests H3.

### 8.3 Qubit scaling (N8)

Run Q-S/Q-G at **8, 10, 12 qubits** (per-drug embeddings of 4, 5, 6 dims) against parameter-matched classical heads on the matching backbones.

### 8.4 Metrics

- Primary: **AUPRC, AUROC** (S3)
- Secondary: F1, accuracy, MCC
- **Calibration:** Expected Calibration Error (ECE), reliability diagrams (N13)
- Efficiency: trainable parameters, epochs to convergence
- Simulation cost reported only as a descriptive number, **not** as a speed claim (everything is simulated)

### 8.5 Logging

One row per run in `results/`: arm, setting, dataset, seed, data fraction, qubit count, parameter count, all metrics, config hash.

**Deliverable:** complete results tables (raw, one row per run).

---

## 9. Quantum-specific analysis (N9, N10, N11, N12)

1. **Entanglement ablation (N9):** retrain Q-S with cross-register entanglers removed. Compare against full Q-S.
2. **Entanglement entropy (N9):** compute von Neumann entropy across the A|B cut for trained circuits on test inputs. Does the model use the cross-register entanglement it was given?
3. **Classical simulability (N10):**
   - Train a classical surrogate (e.g. small MLP or low-rank model) to mimic Q-S outputs.
   - Report surrogate fidelity (R², rank agreement) on held-out pairs.
   - Compute the circuit's Fourier/frequency spectrum with respect to inputs.
   - High surrogate fidelity ⇒ the circuit is not exploiting classically hard structure.
4. **Trainability (N11):** gradient variance across random initializations as a function of qubit count and depth; effective dimension / Fisher-information spectrum.
5. **Noise robustness (N12):**
   - Shot noise: 100, 1k, 10k shots.
   - Depolarizing and readout-error noise models.
   - Plot AUROC vs noise level for Q-S, Q-G, Q-K.
6. **Hardware inference (optional, N12):** run the trained Q-S weights on a few hundred test pairs on a real device through a cloud provider (check current free-tier availability). If unavailable, state that and rely on noise simulation.

**Deliverable:** analysis tables and figures for each item.

---

## 10. Statistical analysis (N14)

1. **Confidence intervals:** 95% CIs across seeds for every metric.
2. **Paired tests:** Wilcoxon signed-rank on paired seeds/splits, with **Holm correction** across comparisons.
3. **Equivalence testing:** TOST with the pre-registered margin δ, so a "tie" between quantum and classical is statistically supported (tests H2).
4. **Effect sizes:** report standardized differences with CIs.
5. **Hypothesis verdicts:**
   - H1: compare C-S vs C-G and Q-S vs Q-G.
   - H2: compare Q-S vs C-S and Q-G vs C-G at matched parameters (equivalence).
   - H3: check whether any quantum-vs-classical gap appears only at small data fractions or in S3.
6. Report results for **all arms and all pre-registered comparisons**, including those that do not support the hypotheses.

**Deliverable:** statistics tables with a verdict per hypothesis.

---

## 11. Robustness checks

- Rerun main comparisons with the 1:5 negative ratio.
- Rerun with the structure-only cold-start variant.
- Rerun with the scaffold split (if included).
- Compare frozen-backbone vs end-to-end results.
- Check that conclusions hold on both datasets.

**Deliverable:** robustness table showing which conclusions are stable.

---

## 12. Final deliverables

1. **Results package**
   - 2×2 comparison with CIs (S1/S2/S3, both datasets)
   - Performance vs parameter-count curves
   - Data-efficiency curves
   - Qubit-scaling results
   - Calibration results
2. **Quantum analysis package**
   - Entanglement ablation and entropy
   - Surrogate fidelity and spectrum
   - Trainability plots
   - Noise robustness (and hardware run if done)
3. **Statistics package:** Wilcoxon/Holm and TOST tables, hypothesis verdicts.
4. **Reproducibility package:** code, configs, split indices, cached embeddings, seeds, environment lockfile, one-command scripts to rerun everything.

---

## 13. Timeline

| Weeks | Steps | Milestone |
|---|---|---|
| 1 | 1 | Repo, environment, frozen protocol |
| 2–4 | 2, 3 | Clean data, splits, negatives, leakage tests pass |
| 4–6 | 4 | Heterogeneous graph and features built |
| 6–8 | 5 | Backbone trained, frozen, **gate passed** |
| 8–10 | 6, 7 | All heads implemented, tuned, capacity sweeps configured |
| 10–13 | 8 | Full experiment grid, data-efficiency, qubit scaling |
| 12–14 | 9 | Quantum analysis, noise, optional hardware |
| 14–15 | 10, 11 | Statistics, verdicts, robustness |
| 15–16 | 12 | Final packaging and reproducibility check |

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Backbone underperforms published baselines | Simplify (drop pretraining first) or use a pretrained molecular encoder; do not proceed until the gate passes |
| Quantum training unstable / barren plateaus | Small learning rates, near-identity initialization, shallow depth, gradient-variance monitoring |
| Quantum simulation too slow | Batched PyTorch statevector simulator; subsample for kernel arms |
| Negatives leak popularity | Degree-matched sampling plus degree-only shortcut baseline |
| Quantum and classical tie | Expected; TOST equivalence testing and the simulability analysis make a tie an informative result |
| Frozen backbone biases results toward simple heads | End-to-end confound experiment (Step 5.7) |
| Hardware access unavailable | Rely on noise simulation and say so explicitly |

---

## 15. Definition of done

- [ ] Frozen protocol committed before experiments
- [ ] Leakage tests pass for S1/S2/S3
- [ ] Backbone gate passed
- [ ] Q-S symmetry test passes to 1e-6
- [ ] All six arms plus floors and ablations run on 2 datasets × 3 settings × 10 seeds
- [ ] Capacity sweeps, data-efficiency and qubit-scaling completed
- [ ] Entanglement, surrogate, trainability and noise analyses completed
- [ ] Wilcoxon/Holm and TOST results produced; H1–H3 verdicts recorded
- [ ] Robustness checks completed
- [ ] Everything reproducible from the repo with documented commands
