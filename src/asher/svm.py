"""Reproduce the historical SVM protocol without deleted checkpoint modules.

Uses the original seeds, candidate order, calibration and tie rule. The shared
metric function pools outer folds per repeat. Archives are never training inputs.
"""
from pathlib import Path
import hashlib, importlib.metadata, json, time
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from metrics import select_threshold_target_sensitivity

ROOT=Path(__file__).resolve().parents[2]
GRIDS={'Linear_SVM':{'C':[.01,.1,1,10]},'RBF_SVM':{'C':[.01,.1,1,10],'gamma':[.001,.01,.1,1]}}

def make_svm(name,params):
    base=Pipeline([('scale',StandardScaler()),('svc',SVC(kernel='linear' if name=='Linear_SVM' else 'rbf',class_weight=None,random_state=707,cache_size=256,tol=1e-3,**params))])
    return CalibratedClassifierCV(estimator=base,method='sigmoid',cv=StratifiedKFold(3,shuffle=True,random_state=707),ensemble=False,n_jobs=1)

def write_csv(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');frame.to_csv(temp,index=False);temp.replace(path)

def expected_keys(data,repeats):
    rows=[]
    for repeat in range(repeats):
        for fold,(_,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=707+repeat).split(data,data.target)):
            rows.extend({'repeat':repeat+1,'fold':fold,'id':int(data.iloc[i].id),'y_true':int(data.iloc[i].target)} for i in va)
    return pd.DataFrame(rows)

def validate_predictions(frame,data,name,repeats):
    keys=['repeat','fold','id','y_true']
    if frame.duplicated(['repeat','id']).any() or set(frame.model)!={name}:
        raise ValueError('Duplicate IDs or wrong model labels')
    actual=frame[keys].sort_values(['repeat','id']).reset_index(drop=True)
    expect=expected_keys(data,repeats).sort_values(['repeat','id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expect,check_dtype=False,check_exact=True)
    v=frame[['y_prob','threshold']].to_numpy()
    if not np.isfinite(v).all() or not ((v>=0)&(v<=1)).all():raise ValueError('Invalid probabilities or thresholds')
    if not frame.groupby(['repeat','fold']).threshold.nunique().eq(1).all():raise ValueError('Threshold varies within fold')

def run_shared_svm(data,output,models,repeats=10,outer_folds=5,inner_folds=5,resume=True):
    from final_analysis import summarize_prediction_rows
    if outer_folds!=5 or inner_folds!=5:raise ValueError('SVM protocol requires five inner and outer folds')
    if not 1<=repeats<=10:raise ValueError('Expected 1 to 10 repeats')
    if any(m not in GRIDS for m in models) or len(set(models))!=len(models):raise ValueError('Invalid or duplicated SVM model request')
    data=data.sort_values('id').reset_index(drop=True)
    from data_prep import FEATURE_NAMES
    x=data[FEATURE_NAMES]; y=data.target.to_numpy(dtype=int)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity={'data_sha256':hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'repeats':repeats,'outer_folds':5,'inner_folds':5,'outer_seed':'707+repeat_index','inner_seed':'1707+5*repeat_index+fold','calibration_seed':707,'model_seed':707,'grid':GRIDS,'libraries':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy']}}
    for name in models:
        manifest=output/f'asher_{name}_manifest.json'
        predpath=output/'predictions'/f'{name}.csv'; foldpath=output/f'outer_fold_metrics_{name}.csv'; searchpath=output/f'inner_search_{name}.csv'
        if any(p.exists() for p in [predpath,foldpath,searchpath]) and not manifest.exists():
            raise ValueError('Refusing to overwrite archived or unattributed SVM results; use an empty output directory')
        if manifest.exists():
            if not resume:raise ValueError('Use a new output directory for a fresh run')
            if json.loads(manifest.read_text())!=identity:raise ValueError('Run configuration, source, data or environment changed')
        else:manifest.write_text(json.dumps(identity,indent=2),encoding='utf-8')
        preds=pd.read_csv(predpath).to_dict('records') if predpath.exists() else []
        folds=pd.read_csv(foldpath).to_dict('records') if foldpath.exists() else []
        searches=pd.read_csv(searchpath).to_dict('records') if searchpath.exists() else []
        for rep in range(repeats):
            outer=StratifiedKFold(5,shuffle=True,random_state=707+rep)
            for fold,(tr,va) in enumerate(outer.split(x,y)):
                old=[r for r in preds if r['repeat']==rep+1 and r['fold']==fold]
                oldfold=[r for r in folds if r['repeat']==rep+1 and r['fold']==fold]
                oldsearch=[r for r in searches if r['repeat']==rep+1 and r['fold']==fold]
                if old or oldfold or oldsearch:
                    if len(old)==len(va) and len(oldfold)==1 and len(oldsearch)==len(list(ParameterGrid(GRIDS[name]))):
                        # A corrupt saved fold must fail, not be silently repaired.
                        actual=pd.DataFrame(old)
                        expect=expected_keys(data,repeats).query('repeat==@rep+1 and fold==@fold')
                        pd.testing.assert_frame_equal(actual[['id','y_true']].sort_values('id').reset_index(drop=True),expect[['id','y_true']].sort_values('id').reset_index(drop=True),check_dtype=False)
                        vv=actual[['y_prob','threshold']].to_numpy()
                        if not np.isfinite(vv).all() or not ((vv>=0)&(vv<=1)).all():raise ValueError('Corrupt checkpoint')
                        continue
                    raise ValueError('Incomplete SVM checkpoint bundle; preserve it and restart in an empty output directory')
                start=time.perf_counter(); inner=list(StratifiedKFold(5,shuffle=True,random_state=1707+5*rep+fold).split(x.iloc[tr],y[tr]));best=None; search=[]
                for candidate,params in enumerate(ParameterGrid(GRIDS[name])):
                    oof=np.full(len(tr),np.nan);scores=[]
                    for a,b in inner:
                        fitted=make_svm(name,params).fit(x.iloc[tr[a]],y[tr[a]])
                        oof[b]=fitted.predict_proba(x.iloc[tr[b]])[:,1]
                        scores.append(roc_auc_score(y[tr[b]],oof[b]))
                    score=float(np.mean(scores))
                    search.append(dict(model=name,repeat=rep+1,fold=fold,candidate=candidate,params_json=json.dumps(params,sort_keys=True),mean_inner_auroc=score))
                    if best is None or score>best[0]+1e-12:best=(score,params,oof,candidate)
                score,params,oof,candidate=best
                if not np.isfinite(oof).all():raise ValueError('Incomplete inner OOF predictions')
                threshold,sens=select_threshold_target_sensitivity(y[tr],oof,.95)
                if sens<.95:raise ValueError('Threshold target not met on training OOF')
                fitted=make_svm(name,params).fit(x.iloc[tr],y[tr]);prob=fitted.predict_proba(x.iloc[va])[:,1]
                frame=pd.DataFrame(dict(model=name,repeat=rep+1,fold=fold,id=data.iloc[va].id.to_numpy(),y_true=y[va],y_prob=prob,threshold=threshold))
                preds.extend(frame.to_dict('records'));searches.extend(search)
                folds.append(dict(model=name,repeat=rep+1,fold=fold,n_train=len(tr),n_validation=len(va),params_json=json.dumps(params,sort_keys=True),selected_candidate=candidate,threshold=threshold,inner_sensitivity=sens,seconds=time.perf_counter()-start,**summarize_prediction_rows(frame)))
                write_csv(pd.DataFrame(searches),searchpath);write_csv(pd.DataFrame(folds),foldpath);write_csv(pd.DataFrame(preds),predpath)
                print(f'{name}: repeat {rep+1}/{repeats}, fold {fold+1}/5 complete',flush=True)
        validate_predictions(pd.DataFrame(preds),data,name,repeats)

def compare_archive(output,archive,models):
    rows=[]
    for name in models:
        new=pd.read_csv(Path(output)/'predictions'/f'{name}.csv');old=pd.read_csv(Path(archive)/'predictions'/f'{name}.csv')
        merged=new.merge(old,on=['repeat','fold','id','y_true','model'],validate='one_to_one',suffixes=('_new','_archive'),how='outer',indicator=True)
        if not merged['_merge'].eq('both').all():raise ValueError('Archived prediction keys differ')
        pdiff=np.abs(merged.y_prob_new-merged.y_prob_archive);tdiff=np.abs(merged.threshold_new-merged.threshold_archive)
        rows.append({'model':name,'rows':len(merged),'max_probability_difference':float(pdiff.max()),'max_threshold_difference':float(tdiff.max()),'matches_1e_7':bool((pdiff<=1e-7).all() and (tdiff<=1e-7).all())})
    write_csv(pd.DataFrame(rows),Path(output)/'archive_reproduction_check.csv')
    return rows
