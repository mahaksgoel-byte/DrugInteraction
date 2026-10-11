"""Explain the validation-selected cold-start improvement experiment."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def render(root: Path):
    parent=root/'results/cold_start_improvement'
    runs=sorted(parent.glob('*/protocol.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not runs:return
    st.subheader('Cold-start accuracy: new algorithm experiments')
    names={p.parent.name:p.parent for p in runs}
    directory=names[st.selectbox('Accuracy experiment',list(names),key='accuracy_experiment')]
    st.write('These models tackle S3: predicting pairs where neither drug appeared in training. They use classical algorithms and all training pairs. Their results are separate from the capped classical-versus-quantum comparison.')
    with st.expander('What changed, in everyday language',expanded=True):
        st.markdown('''
        - **Use informative summaries:** the previous drug vector had 3,592 entries. We compare 77 compact values with 165 selected values per drug. These include chemical size, fat solubility, polarity, rings and hydrogen-bond properties calculated from the existing molecular structures. Compression and feature selection use training drugs only.
        - **Learn with boosted trees:** each small decision tree corrects mistakes left by earlier trees. Shallow trees, a minimum of 80 training pairs per leaf and regularization limit memorization. Every 25-tree checkpoint is logged; these are boosting rounds, not neural epochs.
        - **Transfer evidence from similar drugs:** find chemically or clinically similar training drugs and examine their training interactions. Missing side-effect records fall back to chemistry. Held-out interaction labels are never inserted into this model.
        - **Select consistently:** compare six tree configurations and nine similarity configurations on three validation cohorts. Choose one algorithm by mean validation accuracy, breaking ties with AUPRC. Each cohort gets its own validation-selected threshold. The full test folds are scored only after this decision is saved.
        - **Why selection changed:** earlier experiments prioritized AUPRC, a ranking metric. This experiment explicitly prioritizes validation accuracy in response to the accuracy goal. All other metrics remain visible.
        ''')
        st.caption('Repeatedly studied historical test cohorts cannot establish independent confirmation. These runs remain exploratory; no guarantee of 70%, 85%, zero overfitting, or SOTA is implied.')
        st.write('Dataset context: the available eight DDInter category downloads contained 1,939 source drugs. Structure mapping and deduplication produced 1,487 modeled drugs and 120,272 listed interaction pairs. Only 858 drugs have mapped SIDER information and 337 have ClinPGx information. The benchmark samples one unlisted pair per listed pair; an unlisted pair may be an undocumented interaction. Neither severity nor held-out interaction counts are input features.')
        st.markdown('[How histogram gradient boosting works](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)')
    report_file=directory/'ensemble_comparison.json'
    if not report_file.exists():report_file=directory/'comparison.json'
    if report_file.exists():
        report=json.loads(report_file.read_text())
        st.success(f"Completed. Validation-selected algorithm: {report['selection']['model']}")
        st.dataframe(pd.DataFrame([{'S3 cohort':r['seed'],'Test accuracy (%)':100*r['test']['accuracy'],
            'Training accuracy (%)':100*r['train']['accuracy'] if 'train' in r else None,
            'Validation accuracy (%)':100*r['validation']['accuracy'],'Test AUROC':r['test']['auroc'],
            'Test AUPRC':r['test']['auprc'],'Test pairs':r['test']['pairs']} for r in report['results']]),width='stretch',hide_index=True)
        st.caption(f"Mean test accuracy: {100*sum(r['test']['accuracy'] for r in report['results'])/len(report['results']):.2f}%. Cohorts overlap; the mean is descriptive.")
        if 'recipe' in report['selection']:
            st.write('Final selection also considered fixed model averages and tree/similarity blends using validation data only. The selected recipe is shown below, even if the single model wins.')
            st.json(report['selection']['recipe'])
        with st.expander('Model selection and complete report'):
            st.json(report)
            st.download_button('Download accuracy experiment report',report_file.read_text(),file_name=directory.name+'_report.json')
        if any('error_slices' in r for r in report['results']):
            with st.expander('Where the model still makes mistakes'):
                st.dataframe(pd.DataFrame([dict(seed=r['seed'],**g) for r in report['results'] for g in r.get('error_slices',[])]),width='stretch',hide_index=True)
                st.caption('These are descriptive test-set slices after model selection. Groups have different difficulty and class balance, so differences are not proof that one feature causes a gain. No settings are chosen from these slices.')
            st.write('Compare training, validation and test accuracy above. A large gap means the model fits familiar drugs better than unfamiliar ones. Regularization reduces this risk; it cannot guarantee zero overfitting.')
        with st.expander('Demonstrate the improved model on a saved test pair'):
            seed=st.selectbox('Improvement test cohort',[r['seed'] for r in report['results']],key='improvement_replay_seed')
            file_name='ensemble_test_predictions.csv.gz' if 'recipe' in report['selection'] else 'test_predictions.csv.gz'
            prediction_file=directory/f'seed_{seed:02d}'/file_name
            predictions=pd.read_csv(prediction_file)
            row=st.number_input('Improvement test pair row',min_value=0,max_value=len(predictions)-1,value=0,step=1,key=f'improvement_row_{seed}')
            pair=predictions.iloc[int(row)]
            drugs=pd.read_csv(root/'data/processed/drug_table.csv').set_index('drug_id')
            st.write(f"**{drugs.loc[pair.drug_id_a,'drug_names']}** + **{drugs.loc[pair.drug_id_b,'drug_names']}**")
            st.write('Model decision: '+('listed interaction' if pair.prediction==1 else 'unlisted class')+'. Benchmark label: '+('listed interaction' if pair.label==1 else 'sampled unlisted pair')+'.')
            st.caption('Replay of a real held-out prediction. Unlisted does not mean safe; this benchmark does not provide clinical advice. Scores from rank blends are ranking scores, not probabilities.')
            st.download_button('Download improved-model test predictions',prediction_file.read_bytes(),file_name=f'improved_seed_{seed}_predictions.csv.gz')
    else:
        st.info('Experiment is running or incomplete. Refresh to see new progress. No final test score is available until all validation selection finishes.')
    audit=directory/'data_audit.json'
    if audit.exists():
        with st.expander('Dataset coverage and feature audit'):
            st.json(json.loads(audit.read_text()))
            st.write('An unavailable side-effect or gene record does not mean a drug has no side effects or gene associations. Coverage varies across cohorts and can affect prediction quality.')
    novelty=directory/'chemical_novelty.json'
    if novelty.exists():
        with st.expander('How unfamiliar are the validation drugs?'):
            st.dataframe(pd.DataFrame(json.loads(novelty.read_text())),width='stretch',hide_index=True)
            st.caption('Morgan Tanimoto measures overlap between chemical fingerprint bits: 1 means identical fingerprints, 0 means no shared bits. These bins are descriptive, not clinical thresholds. About one quarter of these validation drugs have no training neighbor above 0.3 similarity.')
    with st.expander('Every boosting checkpoint and similarity trial'):
        events=directory/'events.jsonl'
        if events.exists():
            rows=[]
            for line in events.read_text().splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass # A running process may be writing the last line.
            table=[{'Stage':r['stage'],'Seed':r.get('seed'),'Model':r.get('model'),
                    'Boosting round':r.get('round'),'Validation accuracy':r.get('validation',{}).get('accuracy'),
                    'Validation AUPRC':r.get('validation',{}).get('auprc'),'Elapsed seconds':r['elapsed_seconds']}
                   for r in rows if r['stage'] in {'boosting_round','transfer_validation'}]
            st.dataframe(pd.DataFrame(table),width='stretch',hide_index=True)
            st.download_button('Download complete accuracy training log',events.read_text(),file_name=directory.name+'_events.jsonl')
        for folder in sorted(directory.glob('seed_*')):
            sensitivity=folder/'feature_sensitivity.json'
            if sensitivity.exists():
                st.write(f'{folder.name}: validation feature sensitivity')
                st.dataframe(pd.DataFrame(json.loads(sensitivity.read_text())),width='stretch',hide_index=True)
        st.caption('Feature sensitivity shuffles a whole feature group on validation pairs. A larger ranking-score drop means the model relied on that group. Correlated groups and one shuffle limit interpretation; this is not proof of a biological cause.')
    with st.expander('Watch or reproduce this experiment'):
        st.code(f"Get-Content '{events.relative_to(root)}' -Wait",language='powershell')
        st.code('.\\.quantum-venv\\Scripts\\python.exe -m src.backbone.improve_s3 --output-dir results/cold_start_improvement/my_new_run',language='powershell')
        st.code('.\\.quantum-venv\\Scripts\\python.exe -m src.backbone.ensemble_s3 results/cold_start_improvement/my_new_run',language='powershell')
        st.caption('Use an unused output directory. No earlier run is overwritten.')
