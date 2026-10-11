"""Compare fixed ensemble recipes using only saved validation predictions.

Never reads base-run test predictions or metrics to choose an ensemble.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

from src.backbone.improve_s3 import ROOT, metrics, pair_features, transfer


RECIPES_NOTE={'recipes':['best single candidate','mean of all six boosted-tree candidates',
                        'mean of top three boosted-tree candidates by mean validation accuracy',
                        'rank blend of best tree and best transfer at tree weights .25/.5/.75'],
              'selection':'mean validation accuracy; AUPRC tie-break; no test-based choices',
              'rank_transform':'CDF fitted on each model validation scores; same transform applied to test',
              'limits':'Exploratory reuse of validation and historical test cohorts; no independent confirmation'}


def model_scores(bundle,model,indices):
    if model.startswith('boost_'):
        view=model.split('_')[1]
        return bundle['models'][model].predict_proba(pair_features(bundle['views'][view],indices,bundle['similarities']))[:,1]
    cfg=bundle['transfer_configs'][model];w=cfg['side_weight'];sims=bundle['similarities']
    sim=(1-w)*sims[0]+w*sims[1]
    missing=bundle['views']['compact'][:,-2]==0
    sim=np.where(missing[:,None]|missing[None,:],sims[0],sim)
    return transfer(sim,bundle['train_rows'],bundle['adjacency'],cfg['k'])[indices[:,0],indices[:,1]]


def run(directory):
    directory=Path(directory)
    target=directory/'ensemble_comparison.json'
    if target.exists():raise FileExistsError('Ensemble result already exists; preserve it')
    protocol=json.loads((directory/'protocol.json').read_text())
    seeds=protocol['seeds']
    # Only validation metadata is read before selection is frozen.
    summary=pd.read_csv(directory/'validation_summary.csv')
    ranking=summary.groupby('model')[['accuracy','auprc']].mean().sort_values(['accuracy','auprc'],ascending=False)
    best=ranking.index[0];trees=[x for x in ranking.index if x.startswith('boost_')]
    nearest=next(x for x in ranking.index if x.startswith('transfer_'))
    recipes={'best_single':{'models':[best],'weights':[1.],'rank':False},
             'all_trees':{'models':trees,'weights':[1/len(trees)]*len(trees),'rank':False},
             'top_three_trees':{'models':trees[:3],'weights':[1/3]*3,'rank':False}}
    for w in [.25,.5,.75]:recipes[f'tree_transfer_{w}']={'models':[trees[0],nearest],'weights':[w,1-w],'rank':True}
    validation={};rows=[];thresholds={}
    def transform(values,reference):
        return np.searchsorted(np.sort(reference),values,side='right')/len(reference)
    for seed in seeds:
        frame=pd.read_csv(directory/f'seed_{seed:02d}'/'validation_predictions.csv.gz');validation[seed]=frame
        for name,recipe in recipes.items():
            score=sum(w*(transform(frame[model].to_numpy(),frame[model].to_numpy()) if recipe['rank'] else frame[model].to_numpy())
                      for model,w in zip(recipe['models'],recipe['weights']))
            m=metrics(frame.label.to_numpy(),score);rows.append(dict(seed=seed,model=name,**m));thresholds[seed,name]=m['threshold']
    final_ranking=pd.DataFrame(rows).groupby('model')[['accuracy','auprc']].mean().sort_values(['accuracy','auprc'],ascending=False)
    name=final_ranking.index[0];recipe=recipes[name]
    selection={'model':name,'recipe':recipe,'ranking':final_ranking.reset_index().to_dict('records'),'validation_trials':rows}
    (directory/'selection_ensemble_before_test.json').write_text(json.dumps(selection,indent=2))
    print(json.dumps({'selection_frozen':selection}),flush=True)
    results=[]
    for seed in seeds:
        folder=directory/f'seed_{seed:02d}';bundle=joblib.load(folder/'models.joblib')
        # Inspect only the test inputs/labels after the validation-only decision above.
        test=pd.read_csv(ROOT/f'data/splits/S3/seed_{seed:02d}/test.csv.gz')
        lookup={drug:i for i,drug in enumerate(bundle['ids'])}
        ii=np.asarray([[lookup[a],lookup[b]] for a,b in zip(test.drug_id_a,test.drug_id_b)])
        predicted=[]
        for model in recipe['models']:
            score=model_scores(bundle,model,ii)
            if recipe['rank']:score=transform(score,validation[seed][model].to_numpy())
            predicted.append(score)
        score=sum(w*p for w,p in zip(recipe['weights'],predicted));threshold=thresholds[seed,name]
        test['score']=score;test['prediction']=(score>=threshold).astype(int)
        test.to_csv(folder/'ensemble_test_predictions.csv.gz',index=False)
        vm=next({k:v for k,v in r.items() if k not in {'model','seed'}} for r in rows if r['seed']==seed and r['model']==name)
        result=dict(seed=seed,model=name,validation=vm,test=metrics(test.label.to_numpy(),score,threshold))
        # Diagnose generalization without changing the chosen model or cutoff.
        train=pd.read_csv(ROOT/f'data/splits/S3/seed_{seed:02d}/train.csv.gz')
        train_indices=np.asarray([[lookup[a],lookup[b]] for a,b in zip(train.drug_id_a,train.drug_id_b)])
        train_parts=[]
        for model in recipe['models']:
            tp=model_scores(bundle,model,train_indices)
            if recipe['rank']:tp=transform(tp,validation[seed][model].to_numpy())
            train_parts.append(tp)
        train_score=sum(w*p for w,p in zip(recipe['weights'],train_parts))
        result['train']=metrics(train.label.to_numpy(),train_score,threshold)
        presence=bundle['views']['compact'][:,-2]>0
        coverage=presence[ii[:,0]].astype(int)+presence[ii[:,1]].astype(int)
        closest=bundle['similarities'][0][:,bundle['train_rows']].max(1)
        unfamiliar=(closest[ii[:,0]]<.3)|(closest[ii[:,1]]<.3)
        result['error_slices']=[]
        groups=[(f'{n} drugs with SIDER records',coverage==n) for n in (0,1,2)]+[('at least one low-similarity drug',unfamiliar),('both drugs have similarity >=0.3',~unfamiliar)]
        for group,mask in groups:
            if mask.any():result['error_slices'].append({'group':group,'pairs':int(mask.sum()),
                'accuracy':float(accuracy_score(test.label.to_numpy()[mask],score[mask]>=threshold))})
        results.append(result);print(json.dumps(result),flush=True)
    report={'protocol':{**protocol,'ensemble':RECIPES_NOTE},'selection':selection,'results':results}
    target.write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.directory)
