"""Local Streamlit dashboard for DDInter preprocessing and model training."""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.backbone.train_pairs import train  # noqa: E402
from src.backbone.degree_baseline import run as run_degree_baseline  # noqa: E402
from src.data.audit_splits import audit_all  # noqa: E402

st.set_page_config(page_title="Drug Interaction Model Lab", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1400px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: #f5f7fa; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 10px;}
    .plain-note {background:#f1f6fb; border-left:4px solid #4475a8; padding:0.8rem 1rem; border-radius:4px;}
</style>
""", unsafe_allow_html=True)
st.title("Drug Interaction Model Lab")
st.caption("A transparent workspace for preparing the data, training models, and understanding what each result means.")

section = st.sidebar.radio("Sections", ["Preprocessing", "Training", "Classical vs Quantum", "Project roadmap"], index=1)
st.sidebar.caption("Benchmark labels describe listed DDInter pairs. An unlisted pair is unlabeled, not proven safe.")


def artifact_status() -> pd.DataFrame:
    items = [
        ("Raw DDInter tables", ROOT / "data/raw"),
        ("Processed tables", ROOT / "data/processed"),
        ("Pair splits", ROOT / "data/splits"),
        ("Morgan only features", ROOT / "cache/features/morgan_r2_2048_structure_only.npz"),
        ("SIDER only features", ROOT / "cache/features/morgan_r2_2048_sider_only.npz"),
        ("Morgan + SIDER features", ROOT / "cache/features/morgan_r2_2048_morgan_sider.npz"),
        ("Combined Morgan + SIDER + ClinPGx features", ROOT / "cache/features/morgan_r2_2048.npz"),
        ("ClinPGx-only feature ablation", ROOT / "cache/features/morgan_r2_2048_pharmgkb_only.npz"),
        ("ClinPGx clinical gene features", ROOT / "data/raw/pharmgkb/clinicalAnnotations.zip"),
        ("Saved training histories", ROOT / "results/ddinter"),
        ("Thirty-split integrity audit", ROOT / "results/ddinter/split_integrity_audit.json"),
    ]
    rows = []
    for label, path in items:
        exists = path.exists()
        count = len(list(path.rglob("*"))) if exists and path.is_dir() else (1 if exists else 0)
        rows.append({"Artifact": label, "Status": "Available" if exists else "Missing",
                     "Files": count, "Location": str(path.relative_to(ROOT))})
    return pd.DataFrame(rows)


def saved_training_reports() -> list[tuple[Path, dict]]:
    reports = []
    for path in (ROOT / "results" / "ddinter").rglob("training_history.json"):
        try:
            reports.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(reports, key=lambda item: item[0].stat().st_mtime, reverse=True)


def split_setting(report: dict) -> str:
    split_text = str(report.get("split", "")).replace("\\", "/").upper()
    return next((name for name in ("S1", "S2", "S3") if f"/{name}/" in f"/{split_text}/"), "Unknown")


if section == "Preprocessing":
    st.header("Preprocessing")
    st.write("Preprocessing turns the downloaded source files into clean drug records, fair evaluation splits, and model-ready drug descriptions. The table below shows what is already on this computer.")
    st.dataframe(artifact_status(), width="stretch", hide_index=True)
    st.info("The pipeline uses files already in `data/raw`; it does not fetch or replace source data. A missing item may mean its source archive has not been downloaded yet.")
    with st.expander("What the preprocessing pipeline does"):
        st.markdown("""
        1. **Read and clean:** load DDInter interactions and drug metadata; normalize names and identifiers.
        2. **Describe each drug:** create Morgan chemical fingerprints, SIDER side-effect indicators, and available ClinPGx clinical gene/category indicators.
        3. **Separate fairly:** create S1/S2/S3 training, validation, and test files. Test negatives are sampled from legal candidate pairs without using the held-out positive-degree count to select them.
        4. **Save an audit trail:** write processed tables, split metadata, and feature files so runs can be reproduced.

        **Important label caveat:** a pair absent from DDInter is treated as a negative for this benchmark, but that absence does not prove the drugs do not interact. The benchmark is therefore a useful approximation, not a clinical safety test.
        """)
    with st.expander("What these inputs mean, and where they are weak"):
        st.markdown("""
        - **Morgan fingerprint:** 2,048 yes/no bits summarize small chemical patterns. This can represent chemical resemblance, but does not directly say how a drug affects a protein.
        - **SIDER:** up to 1,024 known single-drug side-effect indicators, plus a source-availability flag. Side effects can hint at shared biology, but are indirect and may be missing.
        - **ClinPGx:** up to 512 clinical gene indicators, six category flags and a source flag. The current exact-name map covers 337 of 1,487 drugs. Missing entries mean “no mapped record here,” not “no biology.”
        - **Not present yet:** mapped drug-to-target, enzyme and transporter links. STRING provides protein-to-protein links; it cannot tell us which proteins a drug affects.

        Feature coverage can change how well the model works. The current results do not yet break scores down by how many useful annotations each pair has.
        """)
    st.subheader("Split integrity audit")
    audit_file = ROOT / "results" / "ddinter" / "split_integrity_audit.json"
    if audit_file.exists():
        try:
            audit_report = json.loads(audit_file.read_text(encoding="utf-8"))
            st.success(f"Last audit: {audit_report['split_count']} of {audit_report['split_count']} generated split files passed the integrity checks.")
            st.caption("Checks cover source checksum, unique canonical pairs, no known positive mislabeled as an unlabeled pair, balanced sampling, no pair reuse across folds, and S2/S3 drug-cohort rules.")
        except (OSError, json.JSONDecodeError, KeyError):
            st.warning("An audit file exists but could not be read. Run the audit again to replace it.")
    else:
        st.info("No full split audit has been saved yet.")
    if st.button("Audit all 30 S1/S2/S3 splits"):
        try:
            audit_results = audit_all(ROOT / "data" / "splits", ROOT / "data" / "processed",
                                      ["S1", "S2", "S3"], list(range(10)))
            payload = {"status": "passed", "split_count": len(audit_results), "reports": audit_results}
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            audit_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            st.success(f"All {len(audit_results)} splits passed. The detailed audit is saved under {audit_file.relative_to(ROOT)}.")
            st.dataframe(pd.DataFrame([{"Setting": row["setting"], "Seed": row["split_seed"],
                "Audited pairs": row["pairs_audited"], "Status": row["status"]} for row in audit_results]),
                width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Split audit stopped: {exc}")
    if st.button("Run preprocessing pipeline", type="primary"):
        log_area = st.empty()
        with st.spinner("Running preprocessing and feature generation..."):
            process = subprocess.Popen([sys.executable, "-m", "src.data.pipeline"], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            log = []
            assert process.stdout is not None
            for line in process.stdout:
                log.append(line.rstrip())
                log_area.code("\n".join(log[-80:]), language="text")
            code = process.wait()
        if code == 0:
            st.success("Preprocessing completed.")
            st.rerun()
        else:
            st.error(f"Preprocessing exited with code {code}. See the log above.")

elif section == "Training":
    st.header("Training")
    from src.ui.improvement import render as render_accuracy_improvement
    render_accuracy_improvement(ROOT)
    st.subheader("Neural-model training archive")
    # Show the most recent saved S3 evaluation as a transparent status card.
    saved_reports = saved_training_reports()
    saved_runs = [path for path, _ in saved_reports]
    s3_test_reports = [report for _, report in saved_reports
                       if isinstance(report.get("test_metrics", {}).get("accuracy"), (int, float)) and split_setting(report) == "S3"]
    s3_report = s3_test_reports[0] if s3_test_reports else None
    best_s3_report = max(s3_test_reports, key=lambda report: report["test_metrics"]["accuracy"]) if s3_test_reports else None
    if s3_report:
        accuracy = 100 * s3_report["test_metrics"]["accuracy"]
        best_accuracy = 100 * best_s3_report["test_metrics"]["accuracy"]
        score, best_score, target = st.columns([1, 1, 2])
        score.metric("Latest saved S3 test accuracy", f"{accuracy:.2f}%", help="S3 holds out both drugs in each evaluated pair.")
        best_score.metric("Highest saved neural S3 accuracy", f"{best_accuracy:.2f}%", help="Neural-model archive only. New tree and similarity experiments are reported above. This historical maximum is not used to choose the current model.")
        target.markdown("**What this means**\n\nThe latest strict cold-start result is compared with the 70% project target. A score below that line is a gap to solve, not a reason to tune against test labels. Historical runs may use different cohorts and settings, so the highest score is not a controlled model comparison.")
        st.progress(min(accuracy / 70, 1.0), text=f"{accuracy:.2f}% of the 70% target")
        st.caption(f"Run: {s3_report.get('feature_type', 'features')} · best checkpoint at epoch {s3_report.get('best_epoch')} · {s3_report.get('epochs_completed')} epochs completed · {Path(s3_report.get('feature_file', '')).name}")
        latest_by_setting = {}
        for _, report in saved_reports:
            split_text = str(report.get("split", "")).replace("\\", "/").upper()
            setting_name = next((name for name in ("S1", "S2", "S3") if f"/{name}/" in f"/{split_text}/"), None)
            if setting_name and isinstance(report.get("test_metrics", {}).get("accuracy"), (int, float)) and setting_name not in latest_by_setting:
                latest_by_setting[setting_name] = {
                    "Setting": setting_name,
                    "Split cohort": Path(report.get("split", "")).name,
                    "Test accuracy": f"{100 * report['test_metrics']['accuracy']:.2f}%",
                    "Test AUPRC": f"{report['test_metrics']['auprc']:.4f}",
                    "Features": report.get("feature_type", "unknown"),
                }
        if latest_by_setting:
            st.caption("Latest saved held-out reports by split setting (scores are not directly interchangeable; S1 is the easiest).")
            st.dataframe(pd.DataFrame(list(latest_by_setting.values())), width="stretch", hide_index=True)
        degree_rows = []
        for path in sorted((ROOT / "results" / "ddinter" / "degree_baselines").glob("*.json")):
            try:
                diagnostic = json.loads(path.read_text(encoding="utf-8"))
                degree_rows.append({
                    "Split": f"{split_setting({'split': diagnostic.get('split', '')})} / {Path(diagnostic.get('split', '')).name}",
                    "Test accuracy": f"{100 * diagnostic['test']['accuracy']:.2f}%",
                    "Test AUROC": f"{diagnostic['test']['auroc']:.3f}",
                    "Test AUPRC": f"{diagnostic['test']['auprc']:.3f}",
                    "Test pair endpoints with known training degree": f"{100 * diagnostic['test']['fraction_of_test_endpoints_with_nonzero_train_degree']:.1f}%",
                })
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        if degree_rows:
            with st.expander("Degree-only shortcut audit results"):
                st.markdown("This baseline sees only how many **training positive interactions** each drug had. It does not see chemical structure, side effects or mechanisms.")
                st.dataframe(pd.DataFrame(degree_rows), width="stretch", hide_index=True)
                st.markdown("For seed 2, the degree-only S1/S2 scores are well above chance because known drugs have observable interaction history. In S3, neither test drug has any training degree and the baseline returns chance-level results (50% accuracy, AUROC 0.50). That is the expected cold-start behavior and evidence this simple popularity feature is not leaking held-out interaction degree into S3. The pair/cohort split audit has also passed all 30 generated files. Since the planned graph backbone is not built, graph-message-passing leakage checks do not apply yet.")
        st.markdown("#### Why results from two runs can differ")
        st.markdown("A split seed changes **which drugs** are held out and which unlisted pairs are sampled as benchmark negatives. A new seed is a new test cohort, not another measurement on exactly the same examples. S3 also asks for predictions on two unfamiliar drugs, while S1 allows drug identities to appear in both training and test and is much easier. Feature sources, model settings, selected epoch and validation-chosen threshold can change scores too.")
        st.info("Validation and test contain different pairs. Validation guides checkpoint and setting choices; test is the final check on pairs that did not guide those choices. A validation score can be higher because its held-out drugs happen to be easier for the available features. Multiple held-out cohorts are needed to estimate typical performance.")
        with st.expander("Specific example: why the fresh S3 result changed"):
            st.markdown("""
            In the seed-2 search, four model configurations were compared on validation pairs. The selected width-64/dropout-0.35 model had **68.49% validation accuracy** and **0.7572 validation AUPRC**. Its unseen test cohort scored **63.22% accuracy**, **0.6668 AUROC** and **0.6678 AUPRC**. That is a **5.27 percentage-point** accuracy gap between different sets of drug pairs.

            This does not mean one specific bug caused the gap. Validation and test contain different held-out drugs; their sampled unlisted pairs differ; and choosing the best of four settings can make validation somewhat optimistic. The validation threshold is also chosen on validation examples and then used unchanged on test. Seed 1's 66.40% result used another held-out cohort and another model setting, so the 3.18-point difference between test scores cannot be attributed to a single change. Multiple same-protocol cohorts are needed to separate model quality from cohort difficulty.

            The chosen checkpoint was epoch 1 even though training continued for 9 epochs. Later passes reduced training error without improving the validation-selected ranking score, so more epochs did not fix this result.
            """)
    else:
        st.info("No saved S3 test result found yet. The first run will create one.")
    with st.expander("Algorithm and evaluation guide", expanded=True):
        st.markdown("""
        ### How a prediction is made
        A drug is represented by (1) bits that summarize its chemical building blocks (Morgan fingerprint), (2) known side-effect indicators (SIDER), and (3) clinical gene/category annotations (ClinPGx), when available. A shared neural network learns a compact summary for each drug. The two summaries are combined in a symmetric way, so switching the order of the drugs cannot change the answer.

        ### Training and safeguards
        The model sees small batches of benchmark-labeled pairs and adjusts its weights to reduce binary cross-entropy error using AdamW. **One epoch** means one pass through all training examples. After every epoch it is checked on a separate validation set. The saved checkpoint has the best validation AUPRC: that score measures ranking without choosing a probability cutoff first. The classification cutoff is then chosen on validation data and held fixed for test scoring. The learning rate can decrease when validation ranking stalls; early stopping ends unhelpful training. A test set is for the final report, not for choosing features or settings.

        ### What S1, S2 and S3 mean
        | Setting | In plain language | What it can tell us |
        |---|---|---|
        | **S1 — familiar drugs** | Drug identities can appear in both training and test pairs. | Can the model recognize combinations of drugs it has encountered? This is the easiest setting. |
        | **S2 — one new drug** | Each test pair has one drug held out from training and one familiar drug. | Can the model generalize when one partner is new? |
        | **S3 — two new drugs** | Both drugs in each test pair are from held-out cohorts. | Can the model generalize to pairs of unseen drugs? This is the strictest and most relevant cold-start check. |

        ### Reading the numbers
        Accuracy is the fraction of benchmark pairs classified correctly at the chosen threshold. AUROC measures how often a positive pair ranks above a negative; AUPRC focuses more on finding positive pairs. The **threshold** is the probability cutoff for saying “interaction”; it is chosen on validation data. **Loss** measures how wrong probability estimates are and penalizes confident mistakes. An epoch is a complete training pass. If training loss keeps falling while validation results stall or worsen, the model may be memorizing (overfitting). The benchmark uses a 1:1 sample of **unlabeled** pairs as negatives, so its accuracy is not expected accuracy on real-world drug pairs and is not a clinical safety probability. This model predicts whether a pair is listed as interacting; it does not predict interaction type, severity, dose, or patient-specific risk.
        """)

    with st.expander("What each training control does"):
        st.markdown("""
        | Control | Plain-language meaning | If poorly chosen |
        |---|---|---|
        | **Hidden width** | Number of learned values in each drug summary. | Too large can memorize; too small can miss useful patterns. |
        | **Learning rate** | Size of each weight-adjustment step. | Too high can jump past a useful solution; too low learns slowly. |
        | **Dropout** | Randomly hides some internal signals while learning. | Too little may allow memorizing; too much can hide useful information. |
        | **Weight decay** | Discourages unnecessarily large model weights. | Too little weakens regularization; too much can make the model underfit. |
        | **Batch size** | Number of pairs used for one weight update. | Affects update noise and memory use; it is not the number of epochs. |
        | **Patience** | Number of unhelpful validation epochs allowed before stopping. | Too short may stop early; too long wastes time and can deepen overfitting. |
        | **Maximum epochs** | Hard upper limit on full passes through training pairs. | It is a ceiling, not a target; the saved checkpoint can be much earlier. |
        """)

    with st.expander("What one saved epoch row means"):
        st.markdown("""
        | Saved value | What it measures | How to read it |
        |---|---|---|
        | **Epoch** | One complete pass over the training examples. | The checkpoint can come from any earlier epoch. |
        | **Train loss** | Model error on examples it learned from. | Lower is usually better here, but falling alone does not prove generalization. |
        | **Validation loss** | Error on a separate set used to guide training. | If it rises while train loss falls, the model may be overfitting. |
        | **Validation AUPRC / AUROC** | How well scores rank positive pairs over negative-labeled pairs. | Higher is better; AUPRC is especially useful when positives are uncommon. |
        | **Validation accuracy** | Correct classifications at the validation-chosen cutoff. | Depends on cutoff and the benchmark’s sampled class balance. |
        | **Matthews correlation (MCC)** | A balanced score using all four kinds of result: correct positives/negatives and missed/false alarms. | -1 is fully wrong, 0 is no useful association, +1 is perfect. |
        | **Brier score** | Average squared difference between probability and benchmark label. | Lower is better; it also penalizes confident wrong guesses. |
        | **10-bin calibration error** | Groups predictions by confidence and compares confidence with the fraction labeled positive. | Lower means confidence better matches these benchmark labels; small bins make this estimate approximate. |
        | **Decision threshold** | Probability cutoff used to call a pair positive. | Chosen on validation, then frozen before test scoring. |
        | **Learning rate** | Size of the optimizer's weight update. | It can step down when validation ranking stops improving. |
        | **Elapsed seconds** | Time spent training through that epoch. | Describes this CPU/GPU run; it is not a quality metric. |

        The saved checkpoint is chosen by validation AUPRC. “Best epoch” therefore means the epoch with the best validation ranking score, not necessarily the epoch with the highest accuracy or the lowest training loss.
        """)

    with st.expander("Why S3 is the hard part—and what could move it"):
        st.markdown("""
        In S3, the model has never seen either drug in a test pair. A chemical fingerprint can say two drugs look alike, but it cannot reliably reveal every biological mechanism they share. The ClinPGx feature map currently covers only 337 of 1,487 drugs, and the available STRING data connect proteins to other proteins; they do not say which proteins a drug affects.

        **Most promising next steps:** add licensed, well-mapped drug-to-target, enzyme, and transporter links; measure whether those links actually improve held-out predictions; handle unlisted pairs as uncertain labels (positive-unlabeled learning) rather than treating every absence as a confirmed “no”; and report averages across several drug-held-out splits. More epochs or a larger neural network cannot invent missing mechanism evidence.
        """)

    test_reports = [(path, report) for path, report in saved_reports if report.get("test_metrics")]
    if test_reports:
        st.subheader("Compare saved test runs")
        report_labels = [f"{path.parent.relative_to(ROOT)} | {split_setting(report)} | {report.get('feature_type', 'features')}" for path, report in test_reports]
        default_indices = list(range(min(2, len(report_labels))))
        selected_labels = st.multiselect("Choose completed runs", report_labels, default=[report_labels[i] for i in default_indices], help="Choose more than one run to compare its saved held-out results. Different settings or seeds are not controlled, one-factor comparisons.")
        label_to_report = {label: pair for label, pair in zip(report_labels, test_reports)}
        selected_reports = [label_to_report[label] for label in selected_labels]
        if selected_reports:
            comparison_rows = []
            for path, report in selected_reports:
                metrics = report["test_metrics"]
                config = report.get("config", {})
                val = report.get("validation_metrics_at_best", {})
                comparison_rows.append({
                    "Run": str(path.parent.relative_to(ROOT)),
                    "Setting": split_setting(report),
                    "Drug cohort": Path(report.get("split", "")).name,
                    "Feature sources": report.get("feature_type", "unknown"),
                    "Model seed": report.get("seed", "unknown"),
                    "Width": config.get("width", "unknown"),
                    "Dropout": config.get("dropout", "unknown"),
                    "Best epoch / epochs run": f"{report.get('best_epoch', '?')} / {report.get('epochs_completed', '?')}",
                    "Validation accuracy": f"{100 * val['accuracy']:.2f}%" if "accuracy" in val else "not recorded",
                    "Test accuracy": f"{100 * metrics['accuracy']:.2f}%" if "accuracy" in metrics else "not recorded",
                    "Test AUROC": f"{metrics['auroc']:.4f}" if "auroc" in metrics else "not recorded",
                    "Test AUPRC": f"{metrics['auprc']:.4f}" if "auprc" in metrics else "not recorded",
                    "Test MCC": f"{metrics['matthews_correlation']:.4f}" if "matthews_correlation" in metrics else "not recorded",
                    "Test Brier": f"{metrics['brier_score']:.4f}" if "brier_score" in metrics else "not recorded",
                    "Test calibration error": f"{metrics['expected_calibration_error_10_bins']:.4f}" if "expected_calibration_error_10_bins" in metrics else "not recorded",
                })
            st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
            if len(selected_reports) > 1:
                chosen = [report for _, report in selected_reports]
                reasons = []
                if len({split_setting(r) for r in chosen}) > 1:
                    reasons.append("These runs use different split settings (S1/S2/S3), so the difficulty is different.")
                if len({str(r.get("split", "")) for r in chosen}) > 1:
                    reasons.append("They use different drug cohorts or sampled pairs. A seed can change which compounds are held out, so this is a new test population.")
                if len({r.get("feature_type") for r in chosen}) > 1:
                    reasons.append("They give the model different input information, so feature coverage and missing annotations can change the score.")
                if len({json.dumps(r.get("config", {}), sort_keys=True) for r in chosen}) > 1:
                    reasons.append("Their model settings differ (for example width, dropout, learning rate or batch size). The model can fit patterns differently.")
                if not reasons:
                    reasons.append("These runs appear to share the same split and settings. Remaining differences can come from random weight initialization or nondeterministic hardware operations.")
                st.markdown("**Why their scores may differ:**\n\n" + "\n\n".join(f"- {reason}" for reason in reasons))
                st.caption("A table comparison is descriptive, not proof that one change caused a difference. To test one feature or setting, compare runs that keep the same drug cohort, split, training seed and all other settings.")

    feature_choice = st.selectbox("Drug information", ["Morgan + SIDER + ClinPGx", "Morgan + SIDER", "Morgan only", "SIDER only", "ClinPGx only"],
        help="Compare chemical structure, SIDER side-effect indicators, and exactly mapped ClinPGx clinical gene annotations.")
    feature_file = ROOT / "cache/features/morgan_r2_2048.npz"
    if feature_choice == "Morgan + SIDER":
        feature_file = ROOT / "cache/features/morgan_r2_2048_morgan_sider.npz"
    elif feature_choice == "Morgan only":
        feature_file = ROOT / "cache/features/morgan_r2_2048_structure_only.npz"
    elif feature_choice == "SIDER only":
        feature_file = ROOT / "cache/features/morgan_r2_2048_sider_only.npz"
    elif feature_choice == "ClinPGx only":
        feature_file = ROOT / "cache/features/morgan_r2_2048_pharmgkb_only.npz"
    if not feature_file.exists():
        st.error("Selected features are missing. Open Preprocessing and run the pipeline.")
        st.stop()
    left, right = st.columns(2)
    with left:
        setting = st.selectbox("Evaluation setting", ["S3", "S2", "S1"],
                               help="S3 = both drugs unseen; S2 = one unseen; S1 = random pair split.")
        seed = st.selectbox("Split seed", list(range(10)), index=0)
        epochs = st.slider("Maximum epochs", 10, 200, 100, 10)
        patience = st.slider("Early-stop patience", 3, 30, 12)
    with right:
        batch_size = st.select_slider("Batch size", options=[128, 256, 512, 1024], value=512)
        widths = st.multiselect("Model size (hidden width)", [64, 128, 256], default=[64, 128], help="Smaller models can be less prone to memorizing sparse training data.")
        learning_rates = st.multiselect("Learning rates", [0.0003, 0.001, 0.003], default=[0.0003, 0.001])
        dropouts = st.multiselect("Dropout values to compare", [0.1, 0.2, 0.3, 0.4], default=[0.2, 0.3], help="Randomly masks hidden signals during training to reduce memorization.")
        weight_decays = st.multiselect("Weight decay values to compare", [0.001, 0.01, 0.05], default=[0.01], help="Penalizes unnecessarily large model weights; stronger values add regularization.")

    if not widths or not learning_rates or not dropouts or not weight_decays:
        st.warning("Choose at least one option in each model-setting list.")
        st.stop()

    st.subheader("Run a validation search")
    st.caption("Every trial is ranked by validation AUPRC. Its epochs are saved as they run. Test labels are not used to choose a winning setting.")
    with st.expander("Watch live epoch output in a PowerShell terminal"):
        st.markdown("When the dashboard server is running in a terminal, its window also prints each epoch. To follow the most recently updated JSONL epoch log in a second PowerShell window, run:")
        st.code("$latest = Get-ChildItem results/ddinter -Filter epoch_metrics.jsonl -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1\nGet-Content $latest.FullName -Wait", language="powershell")
    with st.expander("Run the degree-only shortcut check"):
        st.markdown("This tiny diagnostic asks whether simply knowing how often each drug appeared in **training positive pairs** can predict the label. It does not use chemistry. In S3, held-out drugs should have no training interaction degree, so a strong S3 score would suggest an unintended popularity shortcut or a split problem. The threshold is picked on validation and the test result is for diagnosis only.")
        if st.button("Run degree-only check for this split"):
            degree_path = ROOT / "results" / "ddinter" / "degree_baselines" / f"{setting.lower()}_seed_{seed:02d}.json"
            split_dir = ROOT / "data" / "splits" / setting / f"seed_{seed:02d}"
            degree_result = run_degree_baseline(split_dir, degree_path)
            st.success(f"Saved diagnostic: {degree_path.relative_to(ROOT)}")
            st.json(degree_result)
    if st.button("Run validation search and final test", type="primary"):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = ROOT / "results" / "ddinter" / f"{setting.lower()}_seed_{seed:02d}_{run_id}"
        split_dir = ROOT / "data" / "splits" / setting / f"seed_{seed:02d}"
        combos = list(itertools.product(widths, learning_rates, dropouts, weight_decays))
        progress, status = st.progress(0.0), st.empty()
        records, best = [], None
        for idx, (width, lr, dropout, weight_decay) in enumerate(combos, start=1):
            trial_dir = base / f"trial_{idx:02d}_w{width}_lr{str(lr).replace('.', 'p')}_d{str(dropout).replace('.', 'p')}_wd{str(weight_decay).replace('.', 'p')}"
            status.info(f"Validation trial {idx}/{len(combos)} | width={width} | learning rate={lr} | dropout={dropout} | weight decay={weight_decay}")
            epoch_status, epoch_loss_chart, epoch_metric_chart, epoch_trace = st.empty(), st.empty(), st.empty(), []

            def show_epoch(record):
                epoch_trace.append(record)
                epoch_status.caption(f"Epoch {record['epoch']}: train loss {record['train_loss']:.4f} | val loss {record.get('val_loss', float('nan')):.4f} | val AUPRC {record['val_auprc']:.4f} | val accuracy {record['val_accuracy']:.3f}")
                trace = pd.DataFrame(epoch_trace).set_index("epoch")
                if "val_loss" in trace:
                    epoch_loss_chart.line_chart(trace[["train_loss", "val_loss"]])
                epoch_metric_chart.line_chart(trace[["val_auprc", "val_accuracy"]])

            trial = train(split_dir, feature_file, trial_dir, seed=seed, epochs=epochs, patience=patience,
                          batch_size=batch_size, learning_rate=lr, width=width, dropout=dropout,
                          weight_decay=weight_decay, evaluate_test=False, on_epoch=show_epoch)
            summary = {"trial": idx, "width": width, "learning_rate": lr, "dropout": dropout, "weight_decay": weight_decay,
                       "best_epoch": trial["best_epoch"], "validation_auprc": trial["best_validation_auprc"],
                       "validation_auroc": trial["validation_metrics_at_best"]["auroc"],
                       "validation_accuracy": trial["validation_metrics_at_best"]["accuracy"]}
            records.append(summary)
            if best is None or summary["validation_auprc"] > best["validation_auprc"]:
                best = summary
            progress.progress(idx / (len(combos) + 1))

        status.info(f"Selected on validation: width={best['width']}, learning rate={best['learning_rate']}, dropout={best['dropout']}, weight decay={best['weight_decay']}. Fitting and scoring the final held-out test now.")
        final_trace, final_status, final_loss_chart, final_metric_chart = [], st.empty(), st.empty(), st.empty()

        def show_final_epoch(record):
            final_trace.append(record)
            final_status.caption(f"Final fit epoch {record['epoch']}: train loss {record['train_loss']:.4f} | val loss {record.get('val_loss', float('nan')):.4f} | val AUPRC {record['val_auprc']:.4f} | val accuracy {record['val_accuracy']:.3f}")
            trace = pd.DataFrame(final_trace).set_index("epoch")
            if "val_loss" in trace:
                final_loss_chart.line_chart(trace[["train_loss", "val_loss"]])
            final_metric_chart.line_chart(trace[["val_auprc", "val_accuracy"]])

        final = train(split_dir, feature_file, base / "final", seed=seed, epochs=epochs, patience=patience,
                      batch_size=batch_size, learning_rate=best["learning_rate"], width=best["width"],
                      dropout=best["dropout"], weight_decay=best["weight_decay"], evaluate_test=True, on_epoch=show_final_epoch)
        pd.DataFrame(records).to_csv(base / "search_summary.csv", index=False)
        (base / "run_config.json").write_text(json.dumps({"setting": setting, "seed": seed,
            "feature_choice": feature_choice, "epochs_limit": epochs, "patience": patience,
            "batch_size": batch_size, "dropouts": dropouts, "weight_decays": weight_decays,
            "trials": records, "selected": best}, indent=2) + "\n", encoding="utf-8")
        progress.progress(1.0)
        status.success(f"Completed. Every trial and final epoch is saved under {base.relative_to(ROOT)}")
        st.subheader("Validation search results")
        st.caption("These validation scores select a configuration. The final held-out test score below is reported once for this run.")
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
        st.subheader(f"Held-out test result (selected checkpoint: epoch {final['best_epoch']})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{100 * final['test_metrics']['accuracy']:.2f}%")
        c2.metric("Test AUPRC", f"{final['test_metrics']['auprc']:.4f}")
        c3.metric("Test AUROC", f"{final['test_metrics']['auroc']:.4f}")
        c4.metric("Checkpoint epoch / completed", f"{final['best_epoch']} / {final['epochs_completed']}")
        st.markdown("**Extra score checks**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Matthews correlation", f"{final['test_metrics']['matthews_correlation']:.4f}", help="A balanced summary from -1 (wrong ranking of every pair), through 0 (no association), to +1 (perfect predictions).")
        m2.metric("Brier score", f"{final['test_metrics']['brier_score']:.4f}", help="Average squared gap between predicted probability and label; lower is better. It depends on the sampled benchmark labels.")
        m3.metric("Calibration error (10 bins)", f"{final['test_metrics']['expected_calibration_error_10_bins']:.4f}", help="Average mismatch between stated confidence and observed fraction in ten probability bands; lower is better, but bins make it approximate.")
        st.caption("Calibration describes confidence on this sampled benchmark. It is not calibration to real-world DDI prevalence.")
        st.json({k: v for k, v in final.items() if k != "history"})

    st.divider()
    st.subheader("Complete epoch archive")
    histories = list((ROOT / "results" / "ddinter").rglob("training_history.json"))
    histories.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not histories:
        st.info("No saved runs yet. Start a validation search to create epoch records.")
    else:
        choices = {f"{p.parent.relative_to(ROOT)} (updated {datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})": p for p in histories}
        selected_run = st.selectbox("Saved run or trial", list(choices))
        history_path = choices[selected_run]
        saved = json.loads(history_path.read_text(encoding="utf-8"))
        st.caption(f"{saved.get('feature_type', 'Features')} | {saved.get('epochs_completed')} epochs | best epoch {saved.get('best_epoch')} | checkpoint selection uses validation AUPRC")
        hist = pd.DataFrame(saved.get("history", []))
        if not hist.empty:
            metric_columns = [c for c in ["train_loss", "val_loss", "val_auprc", "val_auroc", "val_accuracy", "val_accuracy_at_0_5", "val_balanced_accuracy", "val_matthews_correlation", "val_brier_score", "val_expected_calibration_error_10_bins", "decision_threshold", "learning_rate", "elapsed_seconds"] if c in hist]
            st.line_chart(hist.set_index("epoch")[[c for c in ["train_loss", "val_loss", "val_auprc", "val_accuracy"] if c in hist]])
            st.dataframe(hist[metric_columns], width="stretch", hide_index=True)
        if saved.get("test_metrics"):
            st.write("Saved held-out test metrics (final model only):")
            st.json(saved["test_metrics"])
        if "expected_calibration_error_10_bins" not in saved.get("test_metrics", {}):
            st.info("This is an older run that predates calibration and Matthews-correlation logging. New runs record those extra measures; the saved scores above are left intact.")

elif section == "Classical vs Quantum":
    st.header("Classical vs Quantum")
    from src.ui.quantum import render as render_quantum_kernels
    render_quantum_kernels(ROOT)
    with st.expander("Earlier four-qubit pilot and its complete epoch archive"):
        st.markdown("A controlled experiment that gives a conventional model and a small Qiskit quantum circuit the same drug-pair data.")
        st.warning("This runs Qiskit's exact statevector simulator on your CPU. It is real quantum-circuit simulation, but it is not a quantum computer or a clinical prediction tool.")
        st.markdown("""
        ### What happens in this experiment
        1. The same selected S1/S2/S3 training cohort is used for both models.
        2. Four compact drug-pair inputs are made from Morgan, SIDER and ClinPGx features. PCA and scaling are fitted on training data only.
        3. A logistic regression and a four-qubit variational circuit train on exactly the same stratified sample. The quantum circuit's eight gate angles are updated with SPSA, a method that needs only two circuit evaluations per update.
        4. Both models choose their classification cutoff using validation pairs. They are then scored on the same held-out test-pair sample.

        **Circuit, in plain language:** four qubits begin in a balanced superposition, receive angles from each drug pair, interact through ZZ and CNOT gates, and are measured. The fraction of odd-parity outcomes becomes the model's interaction score. The classical model is deliberately small and gets the same four inputs; this is a baseline, not a claim that it is the strongest possible classical model.

        **Why cap pair counts?** Exact statevector simulation calculates all 16 state amplitudes for each circuit evaluation. This is manageable for a small demonstration, but expensive across hundreds of thousands of pairs. The UI reports the exact pair counts used. Increase the cap gradually if runtime allows.

        **How to read results:** a fair comparison requires the same split, same sampled rows, same features and same validation-only threshold selection. A higher score on one seed does not prove general superiority. Repeat across several seeds before making a research claim. Balanced accuracy and Matthews correlation supplement raw accuracy; AUROC/AUPRC assess ranking.
        """)
        q_left, q_right = st.columns(2)
        with q_left:
            q_setting = st.selectbox("Shared evaluation setting", ["S3", "S2", "S1"], key="q_setting")
            q_seed = st.selectbox("Shared split seed", list(range(10)), index=0, key="q_seed")
        with q_right:
            q_epochs = st.slider("Quantum training epochs", 2, 20, 8, key="q_epochs")
            q_pair_cap = st.select_slider("Training pair sample cap", [128, 256, 512, 1024, 2048], value=512)
        q_feature_file = ROOT / "cache/features/morgan_r2_2048.npz"
        st.caption(f"Features: Morgan + SIDER + ClinPGx · Quantum runtime: local Qiskit Statevector · test rows: same capped sample for both models")
        if st.button("Train and compare both models", type="primary", key="run_quantum_compare"):
            try:
                from src.backbone.compare_qiskit import train_compare
                q_split = ROOT / "data" / "splits" / q_setting / f"seed_{q_seed:02d}"
                q_out = ROOT / "results" / "ddinter" / f"qiskit_{q_setting.lower()}_seed_{q_seed:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                progress, q_status, q_trace = st.progress(0.0), st.empty(), []
                q_chart = st.empty()

                def show_quantum_epoch(record):
                    q_trace.append(record)
                    progress.progress(len(q_trace) / q_epochs)
                    q_status.caption(f"Quantum epoch {record['epoch']}: validation AUPRC {record['val_auprc']:.4f} · validation accuracy {record['val_accuracy']:.3f} · elapsed {record['elapsed_seconds']:.0f}s")
                    q_chart.line_chart(pd.DataFrame(q_trace).set_index("epoch")[["val_auprc", "val_accuracy"]])

                q_result = train_compare(q_split, q_feature_file, q_out, seed=q_seed,
                                         epochs=q_epochs, pair_cap=q_pair_cap, on_epoch=show_quantum_epoch)
                st.success(f"Training complete. Epoch log and model parameters saved to {q_out.relative_to(ROOT)}")
                st.caption(f"Same pairs: {q_result['train_pairs']} train · {q_result['validation_pairs']} validation · {q_result['test_pairs']} test; quantum checkpoint selected at epoch {q_result['best_epoch']}.")
                st.dataframe(pd.DataFrame([
                    {"Model": "Classical logistic regression", **q_result["classical"]},
                    {"Model": "Qiskit 4-qubit variational circuit", **q_result["quantum"]},
                ]), width="stretch", hide_index=True)
                st.json({"circuit": q_result["circuit"], "limitations": q_result["limitations"],
                         "PCA explained variance": q_result["pca_explained_variance_ratio"]})
            except Exception as exc:
                st.error(f"Qiskit comparison could not run: {exc}")
                st.info("Use the project-local environment and install the pinned quantum requirements. See requirements-quantum.txt.")
        q_histories = sorted((ROOT / "results" / "ddinter").glob("qiskit_*_seed_*/training_history.json"),
                             key=lambda path: path.stat().st_mtime, reverse=True)
        if q_histories:
            st.subheader("Saved quantum comparison runs")
            q_history_path = st.selectbox("Saved run", q_histories, format_func=lambda p: str(p.parent.relative_to(ROOT)), key="q_saved")
            q_saved = json.loads(q_history_path.read_text(encoding="utf-8"))
            st.caption(f"{q_saved['mode']} · {q_saved['epochs_completed']} epochs · {q_saved['train_pairs']} / {q_saved['validation_pairs']} / {q_saved['test_pairs']} train/validation/test pairs")
            q_hist = pd.DataFrame(q_saved["history"])
            if not q_hist.empty and "train_batches" not in q_hist:
                st.info("This is an earlier pilot run with one stochastic optimizer update per epoch. Later runs make a full pass over the capped training sample per epoch, so compare the later run records for the current method.")
            if not q_hist.empty:
                st.dataframe(q_hist, width="stretch", hide_index=True)
            st.dataframe(pd.DataFrame([{"Model": name, **q_saved[name]} for name in ("classical", "quantum")]), width="stretch", hide_index=True)

else:
    st.header("Project roadmap and research status")
    st.warning("The demonstrable local benchmark includes real Qiskit kernel comparisons, saved predictions and diagnostics. The larger research plan still needs a heterogeneous graph backbone, the full repeated experiment matrix and an external benchmark. Current results do not establish quantum superiority or state of the art.")
    quantum_available = __import__("importlib.util", fromlist=["find_spec"]).find_spec("qiskit") is not None
    graph_library_available = __import__("importlib.util", fromlist=["find_spec"]).find_spec("torch_geometric") is not None
    roadmap = pd.DataFrame([
        {"Workstream": "DDInter data cleanup, drug mapping and provenance", "Status": "Built", "What this means": "Raw DDInter and reference files are processed locally; audit and provenance files are present."},
        {"Workstream": "S1 / S2 / S3 pair splits and degree shortcut check", "Status": "30 split audits passed; degree baseline added", "What this means": "All generated splits passed pair/label/cohort integrity checks. A train-positive-degree-only logistic diagnostic is available in Training; repeated model training summaries remain."},
        {"Workstream": "Morgan, SIDER and ClinPGx features", "Status": "Built; coverage is sparse", "What this means": "Feature ablations exist. ClinPGx exact-name mapping currently covers 337 / 1,487 drugs."},
        {"Workstream": "Drug-target / enzyme / transporter knowledge graph", "Status": "Blocked by missing mapped data", "What this means": "Local STRING is protein-to-protein data. It does not identify drug targets; DrugBank target/DDI files are not in the project data."},
        {"Workstream": "Frozen heterogeneous GNN backbone and cached drug embeddings", "Status": "Feature encoder fork built; GNN pending", "What this means": "Kernel heads share the same frozen feature MLP and training-only PCA. This is not a drug/protein graph encoder."},
        {"Workstream": "Classical symmetric pair head", "Status": "Built baseline", "What this means": "The current shared encoder and order-invariant pair combiner are trainable with validation checkpointing."},
        {"Workstream": "Qiskit circuits and matched classical kernels", "Status": "Implemented" if quantum_available else "Install project quantum environment", "What this means": "Eight-qubit fidelity SVM vs nine-trial RBF SVM on shared learned features, with full validation/test folds, logistic floor and entanglement ablation. The earlier four-qubit epoch archive is preserved."},
        {"Workstream": "Full two-dataset × three-setting × ten-seed comparison", "Status": "Not complete", "What this means": "There are DDInter baseline runs, but not all planned classical/quantum arms, DrugBank, or a completed repeated-seed matrix."},
        {"Workstream": "Calibration and extra metrics", "Status": "Added for new runs", "What this means": "New training records include MCC, Brier score and 10-bin calibration error. Older runs cannot be backfilled without saved predictions."},
        {"Workstream": "Quantum diagnostics", "Status": "Implemented for kernel runs", "What this means": "Swap invariance, PSD kernel, register entropy, classical imitation, finite-shot sensitivity, simplified global depolarization and validation-fitted calibration. These are simulations, not hardware measurements."},
        {"Workstream": "Capacity, data-efficiency, qubit scaling and trainability", "Status": "Outstanding", "What this means": "Matched capacity sweeps, low-data curves, 8/10/12-qubit comparison, variational gradient statistics and device-specific noise remain research work."},
        {"Workstream": "Final statistical claims and external validation", "Status": "Not ready", "What this means": "Current seed count and benchmark evidence are insufficient for confidence intervals, equivalence claims or a state-of-the-art claim."},
    ])
    st.dataframe(roadmap, width="stretch", hide_index=True)
    st.markdown("### Why the full plan cannot honestly be marked complete")
    compute_status = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    st.markdown(f"This machine reports Qiskit **{'available' if quantum_available else 'not installed'}**, PyTorch Geometric **{'available' if graph_library_available else 'not installed'}**, and compute **{compute_status}**. More importantly, the mapped drug-target and licensed DrugBank interaction/target files needed for the planned knowledge graph and second benchmark are absent. A quantum head trained on a different representation or without the frozen-backbone comparison would not answer the plan’s central research question.")
    st.markdown("### Best next research steps, in order")
    st.markdown("""
    1. **Obtain and map mechanism data:** under the relevant licences, add drug-target, drug-enzyme, transporter and pathway edges; record source, release, mapping confidence and missingness.
    2. **Build a leakage-safe shared representation:** remove held-out DDI labels from graph construction; train a molecular/heterogeneous backbone on training drugs; freeze embeddings once per split so every head receives exactly the same inputs.
    3. **Establish honest baselines:** compare simple similarity/logistic or tree models, the current pair MLP and published graph baselines on the same cohorts. Use positive-unlabeled or sensitivity analyses for unlisted pairs.
    4. **Only then test quantum claims:** run the preregistered classical/quantum × generic/pair-structured design, match parameter budgets, save all seeds, and include circuit ablations and classical surrogates.
    5. **Report uncertainty and realistic use:** repeated drug-held-out cohorts, confidence intervals, calibration and external/temporal evaluation. Never select configurations against the final test labels.
    """)
    st.markdown("### What current research suggests")
    st.markdown("Modern DDI work uses biomedical knowledge graphs and evaluates new-drug cold starts; recent work also checks whether a model actually uses shared target/enzyme/transporter evidence. That makes mechanism coverage and evidence-use tests a stronger next investment than simply increasing epochs. These papers use different data and splits, so their scores are not directly comparable to this project: [KnowDDI knowledge-subgraph learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10978847/), [DDI graph-learning benchmark](https://arxiv.org/abs/2410.18583), and the recent [ColdDDI preprint](https://arxiv.org/abs/2610.05590).")
    st.caption("The canonical detailed plan remains in plan.md. The table above distinguishes completed code from missing data, dependencies and experiments.")

st.caption("Current model: symmetric neural network over per-drug molecular, side-effect, and optional ClinPGx features. The Qiskit comparison is an experimental local circuit simulation; heterogeneous drug-target graph learning is not implemented.")
