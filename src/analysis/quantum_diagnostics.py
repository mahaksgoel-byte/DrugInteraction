"""Fixed-model noise and classical-surrogate checks on validation pairs.

No configuration is chosen here. Shot noise assumes exact state normalization;
depolarization is an analytic global-state channel, not a device noise model.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from src.backbone.quantum_kernel import ROOT, fidelity_kernel, states, symmetric_features, metric, sha256
from src.heads.classical import SymmetricPairHead


def run(directory):
    directory = Path(directory)
    report = json.loads((directory/'comparison.json').read_text())
    saved = joblib.load(directory/'models.joblib')
    checkpoint = Path(saved['checkpoint'])
    if sha256(checkpoint/'best.pt') != report['checkpoint_sha256']:
        raise ValueError('Encoder checkpoint changed after the experiment.')
    origin = json.loads((checkpoint/'training_history.json').read_text())
    feature_file = Path(origin['feature_file'])
    if not feature_file.is_absolute():
        feature_file = ROOT / feature_file
    if sha256(feature_file) != report['features_sha256']:
        raise ValueError('Feature archive changed after the experiment.')
    ckpt = torch.load(checkpoint/'best.pt', map_location='cpu', weights_only=False)
    model = SymmetricPairHead(ckpt['drug_feature_dim'], ckpt['width'], ckpt['dropout'])
    model.load_state_dict(ckpt['model']); model.eval()
    with np.load(feature_file) as archive:
        ids, features = archive['drug_ids'].astype(str), archive['drug_features'].astype(np.float32)
    with torch.no_grad():
        embedding = model.phi(torch.from_numpy(features)).numpy()
    z = np.tanh(saved['scaler'].transform(saved['pca'].transform(embedding))/2)
    lookup = {x:i for i,x in enumerate(ids)}
    def pairs(frame):
        return np.stack((z[[lookup[x] for x in frame.drug_id_a]],z[[lookup[x] for x in frame.drug_id_b]]),axis=1)
    train = pd.read_csv(directory/'train_pairs.csv.gz')
    val = pd.read_csv(directory/'val_pairs.csv.gz')
    if len(val)>512:
        keep,_=train_test_split(np.arange(len(val)),train_size=512,stratify=val.label,random_state=712)
        val=val.iloc[keep].reset_index(drop=True)
    val.to_csv(directory/'diagnostic_validation_pairs.csv.gz', index=False)
    xt,xv=pairs(train),pairs(val)
    q=saved['selected_models']['Quantum fidelity']
    ts=saved['train_states']; vs=states(xv,q['config']['scale'])
    kt,kv=fidelity_kernel(ts,ts),fidelity_kernel(vs,ts)
    y=val.label.to_numpy(); threshold=q['validation']['threshold']
    clean=q['model'].decision_function(kv)
    def norms(s):
        return np.sqrt((1+np.abs(np.einsum('ij,ij->i',s[:,0].conj(),s[:,1]))**2)/2)
    vn,tn=norms(vs),norms(ts)
    unnormalized=np.clip(kv*vn[:,None]*tn[None,:],0,1)
    noise=[dict(model='ideal',level=0,**metric(y,clean,threshold))]
    rng=np.random.default_rng(712)
    for shots in [100,1000,10000]:
        # Random orientation + overlap measurement is Bernoulli with mean Tr(rho_a rho_b).
        measured=rng.binomial(shots,unnormalized)/shots
        noisy=measured/(vn[:,None]*tn[None,:])
        noise.append(dict(model='finite overlap shots; exact normalization',level=shots,
                          **metric(y,q['model'].decision_function(noisy),threshold)))
    for p in [.01,.05,.1]:
        # Query register state only: rho'=(1-p)rho+p I/256; training reference remains ideal.
        query_norm=np.sqrt((1-p)**2*vn**2+(2*p-p*p)/256)
        noisy=((1-p)*unnormalized+p/256)/(query_norm[:,None]*tn[None,:])
        noise.append(dict(model='global depolarization of query state',level=p,
                          **metric(y,q['model'].decision_function(noisy),threshold)))
    ct,cv=(saved['classic_scaler'].transform(symmetric_features(x)) for x in (xt,xv))
    surrogate=SVR(C=1.0,gamma='scale',epsilon=.01).fit(ct,q['model'].decision_function(kt))
    approx=surrogate.predict(cv)
    # Calibrate each selected arm on the same fixed validation subset; never fit on test.
    heldout=pd.read_csv(directory/'test_predictions.csv.gz')
    calibration=[]
    for arm,val_score in [('Quantum fidelity',clean),
                          ('Classical RBF',saved['selected_models']['Classical RBF']['model'].decision_function(cv))]:
        calibrator=LogisticRegression(C=1000, max_iter=2000).fit(val_score.reshape(-1,1),y)
        probability=calibrator.predict_proba(heldout[arm+'_score'].to_numpy().reshape(-1,1))[:,1]
        bins=[]; ece=0.
        for i in range(10):
            selected=(probability>=i/10)&((probability<(i+1)/10) if i<9 else probability<=1)
            if selected.any():
                confidence=float(probability[selected].mean())
                frequency=float(heldout.label.to_numpy()[selected].mean())
                ece+=selected.mean()*abs(confidence-frequency)
                bins.append(dict(lower=i/10,count=int(selected.sum()),confidence=confidence,observed_fraction=frequency))
        calibration.append(dict(arm=arm,method='Platt scaling fitted on fixed validation subset',
                                fit_pairs=len(y),test_pairs=len(heldout),
                                brier_score=float(brier_score_loss(heldout.label,probability)),
                                ece_10_bins=float(ece),reliability=bins))
    result=dict(validation_pairs=len(val),seed=712,noise=noise,
                calibration=calibration,
                surrogate=dict(model='RBF SVR; fixed C=1, epsilon=.01; trained on quantum train scores',
                               validation_r2=float(r2_score(clean,approx)),
                               validation_rmse=float(np.sqrt(mean_squared_error(clean,approx))),
                               validation_spearman=float(spearmanr(clean,approx).statistic)),
                limitations=['Noise and surrogate checks use a fixed validation subset; calibration is fitted there and scored on full test. No test-driven selection.',
                             'Shot experiment uses one seeded measurement realization and exact normalization.',
                             'Global state depolarization is not gate-specific or device-calibrated noise.',
                             'Calibration uses benchmark class prevalence; probabilities are not clinical risk estimates.',
                             'A good classical surrogate weakens claims of difficult-to-simulate structure; it does not prove equivalence.'])
    (directory/'diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.directory)
