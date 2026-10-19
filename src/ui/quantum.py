"""Explain and operate the frozen-encoder quantum-kernel experiment."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


def render_requested_accuracy_log(root: Path) -> None:
    path = root / 'logs' / 'ui_accuracy_training_log.json'
    if not path.exists():
        return
    log = json.loads(path.read_text(encoding='utf-8'))
    with st.expander('Requested training accuracy log', expanded=True):
        st.caption('Generated example data for UI review. No model was trained for these runs.')
        split = st.selectbox('Split', ['S1', 'S2', 'S3'], key='requested_accuracy_split')
        selected_series = [series for series in log['series'] if series['split'] == split]
        comparison = pd.DataFrame([
            {'Epoch': point['epoch'], 'Model': series['model'],
             'Training accuracy': point['train_accuracy']}
            for series in selected_series for point in series['history']
        ])
        st.line_chart(comparison.pivot(index='Epoch', columns='Model', values='Training accuracy'))
        model = st.selectbox('Model', [series['model'] for series in selected_series], key='requested_accuracy_model')
        series = next(item for item in selected_series if item['model'] == model)
        st.caption(f"Run ID {series['run_id']} | {series['epochs']} epochs")
        hist = pd.DataFrame(series['history'])
        st.line_chart(hist.set_index('epoch')[["train_loss", "val_loss", "val_auprc", "val_accuracy"]])
        metric_columns = ["train_accuracy", "train_loss", "val_loss", "val_auprc", "val_auroc", "val_accuracy", "val_accuracy_at_0_5", "val_balanced_accuracy", "val_matthews_correlation", "val_brier_score", "val_expected_calibration_error_10_bins", "decision_threshold", "learning_rate", "elapsed_seconds"]
        st.dataframe(hist[metric_columns], width='stretch', hide_index=True)
        st.download_button('Download generated log', path.read_text(encoding='utf-8'),
                           file_name=path.name, mime='application/json')


def render(root: Path):
    render_requested_accuracy_log(root)
    st.subheader('Quantum kernels with learned drug summaries')
    st.write('The first pilot discarded most input variation and used tiny test samples. This experiment uses the existing trained drug encoder as a shared starting point, then evaluates every pair in the chosen validation and test folds.')
    st.info('Execution uses real Qiskit circuits on the local exact statevector simulator. Quantum hardware and a quantum speed advantage have not been demonstrated.')
    with st.expander('Understand the experiment', expanded=True):
        st.markdown('''
        **Shared knowledge:** a previously trained neural encoder turns drug features into useful summaries. Its weights are frozen: every new model receives the same summaries. PCA compresses them to four values per drug, fitted using training drugs only. This is a feature encoder, not the planned biomedical graph.

        **Classical RBF model:** a support vector machine learns from similarities between drug pairs. It can model nonlinear patterns and is a stronger comparison than logistic regression alone.

        **Quantum fidelity model:** each drug occupies its own four-qubit register. Repeated encoding and gates connecting the registers produce an eight-qubit state. State overlap measures pair similarity. An SVM learns to separate interactions using these similarities. Averaging both drug orders makes the prediction unchanged when the drugs are swapped.

        **Does entanglement help?** a matched ablation removes the gates connecting the registers. A worse or better result is reported, without choosing which outcome to show.

        **Fair selection:** RBF and quantum kernels each receive nine validation trials. Test scores are computed after selecting settings and thresholds. The head training sample is shared; the frozen encoder previously learned from the full training fold. Because encoder and head settings use the same validation cohort, this remains exploratory and needs independent confirmation.

        **Trials versus epochs:** an SVM solves a constrained optimization problem. Its solver iterations and validation trials are logged; these are not neural-network epochs. The earlier variational circuit's epochs remain in the pilot archive below.
        ''')
        st.markdown('[IBM: quantum kernel methods](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/quantum-kernel-methods) explains why overlaps between encoded states can be used as similarities.')
        st.table(pd.DataFrame([
            {'Stage':'Learn drug summaries','Classical':'Frozen neural encoder + PCA','Quantum':'The identical frozen encoder + PCA'},
            {'Stage':'Compare two pairs','Classical':'RBF mathematical similarity','Quantum':'Overlap of simulated eight-qubit circuit states'},
            {'Stage':'Learn the decision','Classical':'SVM optimizer on a normal CPU','Quantum':'SVM optimizer on a normal CPU'},
            {'Stage':'What is quantum?','Classical':'No quantum circuit','Quantum':'RY/RZ encoding, CZ and RZZ gates, state interference and entanglement'},
            {'Stage':'Where does it run?','Classical':'Your computer','Quantum':'Your computer simulates the circuit; no quantum device'}
        ]))
        st.write('S1 tests new pairs involving familiar drugs. S2 tests pairs with one unfamiliar drug. S3 tests pairs where both drugs were absent from training. S3 is harder because memorizing familiar drugs helps less. These completed kernel comparisons use S3.')
        st.write('Accuracy is the fraction of correct yes/no answers at a fixed cutoff. AUROC measures ranking across cutoffs. AUPRC emphasizes finding listed interactions among predicted positives; it is our selection metric. A better ranking can still give worse accuracy at the selected cutoff. More epochs can memorize training examples; early stopping restores the best validation checkpoint, but cannot guarantee zero overfitting.')
    checkpoints = [p.parent for p in (root / 'results/ddinter').rglob('training_history.json')
                   if (p.parent / 'best.pt').exists()]
    choices = [p for p in checkpoints if p.name in {
        's3_seed_02_regularization_final', 's3_seed_01_multimodal_clinpgx', 's3_seed_00_quantum_shared_encoder'}]
    choices.sort(key=lambda p: p.name)
    if choices:
        checkpoint = st.selectbox('Frozen encoder checkpoint', choices,
                                  format_func=lambda p: p.name, key='kernel_checkpoint')
        cap = st.select_slider('Shared head training pairs', options=[512,1024,2048], value=1024, key='kernel_cap')
        st.caption('Validation and test use their entire saved fold. The cap applies only to fitting the new heads. Both arms share the encoder trained on all original training pairs.')
        if st.button('Run controlled kernel comparison', key='run_kernel'):
            try:
                from threadpoolctl import threadpool_limits
                from src.backbone.quantum_kernel import run
                output = root / 'results/quantum_kernels' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                logs, status = [], st.empty()
                def update(row):
                    logs.append(row)
                    status.code('\n'.join(json.dumps(x) for x in logs[-8:]), language='json')
                with threadpool_limits(limits=4):
                    run(checkpoint, output, cap, update)
                st.success(f'Finished. Complete results: {output.relative_to(root)}')
            except Exception as exc:
                st.error(f'Comparison stopped: {exc}')
    reports = sorted((root / 'results/quantum_kernels').glob('*/comparison.json'),
                     key=lambda p:p.stat().st_mtime, reverse=True)
    if reports:
        report_by_name={p.parent.name:p for p in reports}
        selected_name = st.selectbox('Completed kernel experiment', list(report_by_name), key='kernel_report')
        selected = report_by_name[selected_name]
        report = json.loads(selected.read_text())
        cohorts={}
        for path in reports:
            other=json.loads(path.read_text())
            if other['training_pairs']!=report['training_pairs'] or other['features_sha256']!=report['features_sha256']:
                continue
            key=other['split_hashes']['test']
            if key in cohorts:
                continue
            by_arm={r['arm']:r['test'] for r in other['results']}
            cohorts[key]={'Cohort':Path(other['split']).name,
                          'Quantum accuracy (%)':100*by_arm['Quantum fidelity']['accuracy'],
                          'RBF accuracy (%)':100*by_arm['Classical RBF']['accuracy'],
                          'Logistic accuracy (%)':100*by_arm['Classical logistic']['accuracy'],
                          'Quantum minus RBF (points)':100*(by_arm['Quantum fidelity']['accuracy']-by_arm['Classical RBF']['accuracy'])}
        if len(cohorts)>1:
            st.markdown('**Across completed cohorts using the same feature archive and head training size**')
            summary=pd.DataFrame(cohorts.values())
            st.dataframe(summary,width='stretch',hide_index=True)
            st.caption(f"Mean quantum accuracy: {summary['Quantum accuracy (%)'].mean():.2f}% · mean RBF: {summary['RBF accuracy (%)'].mean():.2f}% · mean logistic: {summary['Logistic accuracy (%)'].mean():.2f}%. Latest completed run per cohort is shown, never the highest test score. Encoders were selected separately on their own validation folds. Cohorts overlap, so these descriptive means do not establish statistical significance.")
        st.caption(f"{report['split']} | {report['training_pairs']:,} head training pairs | {report['validation_pairs']:,} validation | {report['test_pairs']:,} test")
        rows = [dict(Model=r['arm'], **{'Test accuracy (%)':100*r['test']['accuracy'],
                'Test AUROC':r['test']['auroc'], 'Test AUPRC':r['test']['auprc'],
                'Test MCC':r['test']['mcc'], 'Validation AUPRC':r['validation']['auprc']}) for r in report['results']]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        indexed = {r['arm']:r for r in report['results']}
        delta = 100*(indexed['Quantum fidelity']['test']['accuracy']-indexed['Classical RBF']['test']['accuracy'])
        st.write(f'Quantum minus RBF accuracy: **{delta:+.2f} percentage points** on this cohort. A positive difference on one cohort does not establish a general quantum advantage.')
        with st.expander('Demonstrate a saved held-out prediction'):
            prediction_file=selected.parent/'test_predictions.csv.gz'
            predictions=pd.read_csv(prediction_file)
            row_index=st.number_input('Test pair row',min_value=0,max_value=len(predictions)-1,value=0,step=1,key=f'pair_{selected.parent.name}')
            pair=predictions.iloc[int(row_index)]
            st.write(f"Drug A: **{pair['drug_id_a']}**; Drug B: **{pair['drug_id_b']}**")
            st.write('Benchmark label: '+('listed DDInter interaction' if pair['label']==1 else 'sampled unlisted pair (not a confirmed safe pair)'))
            st.dataframe(pd.DataFrame([{'Model':r['arm'],
                'Decision':'interaction' if pair[r['arm']+'_prediction']==1 else 'unlisted class',
                'Matches benchmark label':bool(pair[r['arm']+'_prediction']==pair['label']),
                'Raw decision score':pair[r['arm']+'_score'],
                'Validation-selected cutoff':r['validation']['threshold']} for r in report['results']]),width='stretch',hide_index=True)
            st.caption('This replays real saved test predictions. Raw decision scores are not probabilities. The benchmark predicts whether a pair is listed, not severity, dose guidance, or individual patient risk.')
            st.download_button('Download all held-out predictions',prediction_file.read_bytes(),file_name=selected.parent.name+'_test_predictions.csv.gz')
        st.write(f"The four retained components explain {100*report['pca_variance']:.1f}% of variation in the frozen encoder's training-drug summaries. This is not a percentage of raw biological information retained.")
        origin = json.loads((Path(report['checkpoint']) / 'training_history.json').read_text())
        if origin.get('test_metrics',{}).get('accuracy') is not None:
            st.caption(f"Original full-data neural model on this cohort: {100*origin['test_metrics']['accuracy']:.2f}% accuracy. Its prediction head trained on more pairs than these capped heads, so it is a contextual reference, not a matched-data head comparison.")
        with st.expander('Shared encoder training record and experiment provenance'):
            st.write(f"Encoder: {report['checkpoint']}. Best validation checkpoint: epoch {origin.get('best_epoch', 'not recorded')}; epochs completed: {origin.get('epochs_completed', 'not recorded')}.")
            st.json(origin)
            st.json({k:v for k,v in report.items() if k not in {'results','trials','diagnostics'}})
            event_file=selected.parent/'events.jsonl'
            if event_file.exists():
                st.download_button('Download complete run events',event_file.read_text(encoding='utf-8'),file_name=selected.parent.name+'_events.jsonl')
        with st.expander('Circuit checks and complete selection history'):
            diagram=selected.parent/'circuit.txt'
            if diagram.exists():
                st.code(diagram.read_text(encoding='utf-8'),language='text')
            st.json(report['diagnostics'])
            st.write('Swap error checks drug-order invariance. A nonnegative Gram eigenvalue checks that the similarity matrix is valid. Register entropy measures entanglement in bits; it is not an accuracy score.')
            st.dataframe(pd.DataFrame([dict(Model=t['arm'], **t['config'],
                validation_AUPRC=t['validation']['auprc'], solver_iterations=str(t['solver_iterations']))
                for t in report['trials']]), width='stretch', hide_index=True)
            st.json(report['limitations'])
            st.download_button('Download experiment report', selected.read_text(), file_name='quantum_comparison.json')
        diagnostic_file=selected.parent/'diagnostics.json'
        if diagnostic_file.exists():
            diagnostic=json.loads(diagnostic_file.read_text())
            with st.expander('Noise, classical imitation, and confidence calibration'):
                st.write('These checks keep the selected model fixed. Noise and imitation use a fixed validation subset. Calibration is fitted on that validation subset and scored on the full test fold.')
                st.dataframe(pd.DataFrame(diagnostic['noise']),width='stretch',hide_index=True)
                st.json(diagnostic['surrogate'])
                st.write('A high imitation R-squared means an ordinary classical model can reproduce much of the circuit model behavior. It weakens claims that the behavior requires quantum resources.')
                st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k!='reliability'} for r in diagnostic['calibration']]),width='stretch',hide_index=True)
                st.write('Brier score and calibration error measure whether stated confidence matches benchmark outcomes; lower is better. They are not patient safety probabilities. The noise models are simplified simulations, not measurements from a quantum device.')
                st.json(diagnostic['limitations'])
    else:
        st.info('No completed kernel experiment yet. In-progress events are written immediately under results/quantum_kernels.')
    with st.expander('Watch training in your terminal'):
        st.code('.\\.quantum-venv\\Scripts\\python.exe -m src.backbone.quantum_kernel --checkpoint-dir results/ddinter/s3_seed_02_regularization_final --output-dir results/quantum_kernels/my_new_run --cap 1024', language='powershell')
        st.caption('Choose a new output directory for each run. Existing experiments are preserved. The terminal prints each validation trial and simulation progress.')
