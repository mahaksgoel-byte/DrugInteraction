"""Validation-only selection of cold-start tree and similarity-transfer models.

No test CSV is opened until all cohorts' configurations and thresholds are frozen.
Historical S3 test cohorts have previously been reported: this is exploratory.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from src.backbone.quantum_kernel import sha256

ROOT=Path(__file__).resolve().parents[2]


def metrics(y,p,threshold=None):
    if threshold is None:
        fpr,tpr,t=roc_curve(y,p)
        valid=np.isfinite(t)
        threshold=float(t[valid][np.argmax((tpr-fpr)[valid])])
    return dict(accuracy=float(accuracy_score(y,p>=threshold)),auroc=float(roc_auc_score(y,p)),
                auprc=float(average_precision_score(y,p)),threshold=threshold,pairs=len(y))


def tanimoto(x):
    intersection=x@x.T
    size=x.sum(1)
    return intersection/np.maximum(size[:,None]+size[None,:]-intersection,1)


def pair_features(z,indices,similarities):
    a,b=z[indices[:,0]],z[indices[:,1]]
    return np.column_stack([a+b,np.abs(a-b),a*b]+[s[indices[:,0],indices[:,1]] for s in similarities]).astype(np.float32)


def transfer(similarity,train_rows,adjacency,k):
    weights=similarity[:,train_rows].copy()
    # Leave self out even for training nodes: never use a held-out node's labels.
    weights[train_rows,np.arange(len(train_rows))]=0
    top=np.argpartition(weights,-k,axis=1)[:,-k:]
    values=np.take_along_axis(weights,top,axis=1)**2
    values/=np.maximum(values.sum(1,keepdims=True),1e-12)
    w=csr_matrix((values.ravel(),(np.repeat(np.arange(len(weights)),k),top.ravel())),shape=weights.shape)
    return (w@csr_matrix(adjacency)@w.T).toarray()


def run(output,seeds=(0,1,2),max_rounds=200):
    if max_rounds<25 or max_rounds%25:
        raise ValueError('max_rounds must be a positive multiple of 25')
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('Provide distinct split seeds')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    def emit(**row):
        row={'elapsed_seconds':round(time.time()-started,2),**row}
        with (output/'events.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    feature_file=ROOT/'cache/features/morgan_r2_2048.npz'
    protocol={'selection':'mean validation accuracy over all listed cohorts; AUPRC tie-break',
              'seeds':list(seeds),'max_boosting_rounds':max_rounds,'feature_sha256':sha256(feature_file),
              'drug_table_sha256':sha256(ROOT/'data/processed/drug_table.csv'),
              'threshold':'validation Youden J; balanced cohorts make this accuracy-maximizing',
              'test_access':'after every cohort validation trial and final selection file is saved',
              'limits':['Historical test cohorts previously reported; exploratory, not independent confirmation.',
                        'Unlisted pairs are not verified negatives. Transfer treats absent training edges as zero evidence.',
                        'Only train and validation data influence model selection. Cohorts overlap.']}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with np.load(feature_file) as archive:
        ids=archive['drug_ids'].astype(str);raw=archive['drug_features'].astype(np.float32);names=archive['feature_names'].astype(str)
    lookup={v:i for i,v in enumerate(ids)}
    blocks=[np.flatnonzero(np.char.startswith(names,p)) for p in ('morgan_','sider_','pharmgkb_')]
    # Availability indicators are not molecular similarities.
    similarities=[tanimoto(raw[:,cols[~np.char.endswith(names[cols],'source_available')]]) for cols in blocks]
    descriptor_names=['MolWt','MolLogP','TPSA','NumHDonors','NumHAcceptors','NumRotatableBonds',
                      'RingCount','NumAromaticRings','FractionCSP3','HeavyAtomCount','NHOHCount',
                      'NOCount','NumAliphaticRings','NumSaturatedRings','LabuteASA','BalabanJ',
                      'BertzCT','MolMR','NumHeteroatoms']
    drugs=pd.read_csv(ROOT/'data/processed/drug_table.csv').set_index('drug_id')
    descriptors=[]
    for drug in ids:
        mol=Chem.MolFromSmiles(drugs.loc[drug,'canonical_smiles'])
        descriptors.append([float(getattr(Descriptors,name)(mol)) for name in descriptor_names])
    descriptors=np.nan_to_num(np.asarray(descriptors,dtype=np.float32),nan=0,posinf=0,neginf=0)
    audit={'drugs':len(ids),'raw_features':raw.shape[1],'constant_columns':int((raw.var(0)==0).sum()),
           'sider_coverage':int(raw[:,np.where(names=='sider_source_available')[0][0]].sum()),
           'pgx_coverage':int(raw[:,np.where(names=='pharmgkb_source_available')[0][0]].sum()),
           'descriptor_names':descriptor_names,'cohorts':[]}
    bundles={};summaries=[]
    def indices(frame):
        return np.asarray([[lookup[a],lookup[b]] for a,b in zip(frame.drug_id_a,frame.drug_id_b)])
    for seed in seeds:
        folder=output/f'seed_{seed:02d}';folder.mkdir()
        split=ROOT/f'data/splits/S3/seed_{seed:02d}'
        train=pd.read_csv(split/'train.csv.gz');val=pd.read_csv(split/'val.csv.gz')
        ti,vi=indices(train),indices(val);tr=np.unique(ti);vr=np.unique(vi)
        if set(tr)&set(vr):raise ValueError('Expected disjoint S3 train and validation drugs')
        audit['cohorts'].append(dict(seed=seed,train_pairs=len(train),validation_pairs=len(val),train_drugs=len(tr),validation_drugs=len(vr),
            train_sider_coverage=float(raw[tr,3072].mean()),validation_sider_coverage=float(raw[vr,3072].mean()),
            train_pgx_coverage=float(raw[tr,-1].mean()),validation_pgx_coverage=float(raw[vr,-1].mean())))
        parts=[];transforms=[];compact_names=[]
        for cols,n,prefix in zip(blocks,[32,16,8],['chemistry','side_effects','pharmacogenomics']):
            transform=TruncatedSVD(n_components=n,random_state=42).fit(raw[tr][:,cols])
            parts.append(transform.transform(raw[:,cols]));transforms.append(transform)
            compact_names += [f'{prefix}_component_{i}' for i in range(n)]
        # Training-fitted scale keeps products numerically sensible; trees need no normality assumption.
        center=np.median(descriptors[tr],axis=0);scale=np.quantile(descriptors[tr],.75,axis=0)-np.quantile(descriptors[tr],.25,axis=0)
        scaled=(descriptors-center)/np.maximum(scale,1)
        presence=raw[:,[3072,3591]]
        compact=np.column_stack(parts+[scaled,presence]).astype(np.float32)
        compact_names+=descriptor_names+['SIDER_available','ClinPGx_available']
        # Select variable binary inputs using training drugs, without looking at pair labels.
        chosen=np.concatenate([cols[np.argsort(raw[tr][:,cols].var(0))[-n:]] for cols,n in zip(blocks,[64,64,16])])
        views={'compact':compact,'selected':np.column_stack([raw[:,chosen],scaled,presence]).astype(np.float32)}
        feature_names={'compact':compact_names,'selected':list(names[chosen])+descriptor_names+['SIDER_available','ClinPGx_available']}
        candidates={};val_predictions={};models={};curves=[]
        for view,z in views.items():
            xt,xv=pair_features(z,ti,similarities),pair_features(z,vi,similarities)
            for leaves in (7,15,31):
                key=f'boost_{view}_leaves{leaves}'
                model=HistGradientBoostingClassifier(learning_rate=.06,max_leaf_nodes=leaves,min_samples_leaf=80,
                    l2_regularization=10,max_iter=25,early_stopping=False,warm_start=True,random_state=42,max_bins=127)
                best=None;stale=0
                for rounds in range(25,max_rounds+1,25):
                    model.set_params(max_iter=rounds);model.fit(xt,train.label)
                    vp=model.predict_proba(xv)[:,1];m=metrics(val.label.to_numpy(),vp)
                    curve=dict(seed=seed,model=key,round=rounds,validation=m)
                    curves.append(curve);emit(stage='boosting_round',**curve)
                    if best is None or (m['accuracy'],m['auprc'])>(best['validation']['accuracy'],best['validation']['auprc']):
                        best=dict(validation=m,rounds=rounds);models[key]=copy.deepcopy(model);val_predictions[key]=vp;stale=0
                    else:stale+=1
                    if stale>=3:break
                candidates[key]=best
            del xt,xv
        local={i:j for j,i in enumerate(tr)}
        adj=np.zeros((len(tr),len(tr)),dtype=np.float32)
        for a,b in ti[train.label.to_numpy()==1]:adj[local[a],local[b]]=adj[local[b],local[a]]=1
        transfer_configs={}
        for side_weight in (0.,.5,1.):
            sim=(1-side_weight)*similarities[0]+side_weight*similarities[1]
            # A drug with missing side effects falls back to chemistry rather than all-zero neighbors.
            missing=raw[:,3072]==0
            sim=np.where(missing[:,None]|missing[None,:],similarities[0],sim)
            for k in (5,20,50):
                key=f'transfer_side{side_weight}_k{k}'
                score=transfer(sim,tr,adj,k)[vi[:,0],vi[:,1]]
                m=metrics(val.label.to_numpy(),score)
                candidates[key]=dict(validation=m);val_predictions[key]=score
                transfer_configs[key]=dict(side_weight=side_weight,k=k)
                emit(stage='transfer_validation',seed=seed,model=key,validation=m)
        val_out=val.copy()
        for name,score in val_predictions.items():val_out[name]=score
        val_out.to_csv(folder/'validation_predictions.csv.gz',index=False)
        bundle=dict(ids=ids,train_rows=tr,adjacency=adj,views=views,similarities=similarities,models=models,
                    transfer_configs=transfer_configs,candidates=candidates,feature_names=feature_names,
                    transforms=transforms,chosen=chosen,descriptor_center=center,descriptor_scale=scale)
        joblib.dump(bundle,folder/'models.joblib')
        (folder/'validation_trials.json').write_text(json.dumps(candidates,indent=2))
        (folder/'boosting_rounds.json').write_text(json.dumps(curves,indent=2))
        bundles[seed]=bundle
        for name,result in candidates.items():summaries.append(dict(seed=seed,model=name,**result['validation']))
        emit(stage='cohort_validation_complete',seed=seed)
    (output/'data_audit.json').write_text(json.dumps(audit,indent=2))
    summary=pd.DataFrame(summaries);summary.to_csv(output/'validation_summary.csv',index=False)
    ranking=summary.groupby('model')[['accuracy','auprc']].mean().sort_values(['accuracy','auprc'],ascending=False)
    selected=ranking.index[0]
    selection={'model':selected,'ranking':ranking.reset_index().to_dict('records'),
               'thresholds':{str(s):bundles[s]['candidates'][selected]['validation']['threshold'] for s in seeds}}
    (output/'selection_before_test.json').write_text(json.dumps(selection,indent=2))
    emit(stage='selection_frozen',**selection)
    reports=[]
    for seed,bundle in bundles.items():
        split=ROOT/f'data/splits/S3/seed_{seed:02d}';test=pd.read_csv(split/'test.csv.gz');ii=indices(test)
        if set(np.unique(ii))&set(bundle['train_rows']):raise ValueError('Test drug leakage')
        if selected.startswith('boost_'):
            view=selected.split('_')[1];z=bundle['views'][view]
            xx=pair_features(z,ii,similarities);model=bundle['models'][selected];p=model.predict_proba(xx)[:,1]
            # Group permutation on validation only: explanatory sensitivity, not causal attribution.
            val=pd.read_csv(split/'val.csv.gz');iv=indices(val);xv=pair_features(z,iv,similarities)
            baseline=roc_auc_score(val.label,model.predict_proba(xv)[:,1]);importance=[]
            rng=np.random.default_rng(712);dim=z.shape[1]
            base_names=bundle['feature_names'][view]
            for group,cols in [('chemistry',[i for i,n in enumerate(base_names) if n.startswith(('chemistry','morgan_'))]),
                               ('side effects',[i for i,n in enumerate(base_names) if n.startswith(('side_effects','sider_','SIDER'))]),
                               ('pharmacogenomics',[i for i,n in enumerate(base_names) if n.startswith(('pharmacogenomics','pharmgkb_','ClinPGx'))]),
                               ('chemical descriptors',[i for i,n in enumerate(base_names) if n in descriptor_names])]:
                selected_cols=[c+offset*dim for c in cols for offset in range(3)]
                perm=xv.copy();order=rng.permutation(len(perm));perm[:,selected_cols]=perm[order][:,selected_cols]
                importance.append(dict(group=group,validation_auroc_drop=float(baseline-roc_auc_score(val.label,model.predict_proba(perm)[:,1]))))
            (output/f'seed_{seed:02d}'/'feature_sensitivity.json').write_text(json.dumps(importance,indent=2))
        else:
            cfg=bundle['transfer_configs'][selected];w=cfg['side_weight'];sim=(1-w)*similarities[0]+w*similarities[1]
            missing=raw[:,3072]==0;sim=np.where(missing[:,None]|missing[None,:],similarities[0],sim)
            p=transfer(sim,bundle['train_rows'],bundle['adjacency'],cfg['k'])[ii[:,0],ii[:,1]]
        threshold=selection['thresholds'][str(seed)]
        report=dict(seed=seed,model=selected,validation=bundle['candidates'][selected]['validation'],test=metrics(test.label.to_numpy(),p,threshold),
                    split_hashes={fold:sha256(split/f'{fold}.csv.gz') for fold in ['train','val','test']})
        test['score']=p;test['prediction']=(p>=threshold).astype(int)
        test.to_csv(output/f'seed_{seed:02d}'/'test_predictions.csv.gz',index=False)
        reports.append(report);emit(stage='test_result',**report)
    final=dict(protocol=protocol,selection=selection,results=reports,elapsed_seconds=round(time.time()-started,2))
    (output/'comparison.json').write_text(json.dumps(final,indent=2));emit(stage='complete')
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--max-rounds',type=int,default=200)
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.output_dir,args.seeds,args.max_rounds)
