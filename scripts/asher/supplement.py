"""Training-only interpretation supplement; never selects features or edits main results."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];handles=[]
if os.name=='nt' and (Path(sys.prefix)/'Library/bin').is_dir():
    path=str(Path(sys.prefix)/'Library/bin');handles.append(os.add_dll_directory(path));os.environ['PATH']=path+os.pathsep+os.environ.get('PATH','')
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from final_analysis import load_data
from data_prep import FEATURE_NAMES
from asher.svm import make_svm,write_csv

def main():
    data=load_data();x=data[FEATURE_NAMES];y=data.target.to_numpy();out=ROOT/'results/asher/svm_reproduction'
    importance=[];coefs=[]
    for model_name in ['Linear_SVM','RBF_SVM']:
        selected=pd.read_csv(out/f'outer_fold_metrics_{model_name}.csv').set_index(['repeat','fold'])
        for rep in range(10):
            for fold,(tr,_) in enumerate(StratifiedKFold(5,shuffle=True,random_state=707+rep).split(x,y)):
                params=json.loads(selected.loc[(rep+1,fold),'params_json'])
                if model_name=='Linear_SVM':
                    fit=make_svm(model_name,params).fit(x.iloc[tr],y[tr])
                    coef=fit.calibrated_classifiers_[0].estimator.named_steps['svc'].coef_[0]
                    coefs.extend(dict(repeat=rep+1,fold=fold,feature=f,standardized_margin_coefficient=float(c)) for f,c in zip(FEATURE_NAMES,coef))
                if rep==0:
                    a,b=next(StratifiedKFold(5,shuffle=True,random_state=1707+fold).split(x.iloc[tr],y[tr]))
                    fit=make_svm(model_name,params).fit(x.iloc[tr[a]],y[tr[a]])
                    perm=permutation_importance(fit,x.iloc[tr[b]],y[tr[b]],scoring='roc_auc',n_repeats=10,random_state=707+fold,n_jobs=1)
                    importance.extend(dict(model=model_name,repeat=1,fold=fold,feature=f,mean_auroc_drop=float(m),permutation_sd=float(s),n_interpretation=len(b)) for f,m,s in zip(FEATURE_NAMES,perm.importances_mean,perm.importances_std))
        print(model_name+' interpretation complete',flush=True)
    imp=pd.DataFrame(importance);coef=pd.DataFrame(coefs)
    write_csv(imp,out/'training_only_permutation_by_fold.csv')
    summary=imp.groupby(['model','feature']).mean_auroc_drop.agg(['mean','std','min','max']).reset_index().sort_values(['model','mean'],ascending=[True,False])
    write_csv(summary,out/'training_only_permutation_summary.csv')
    write_csv(coef,out/'linear_standardized_margin_coefficients.csv')
    cs=coef.groupby('feature').standardized_margin_coefficient.agg(median='median',q25=lambda v:v.quantile(.25),q75=lambda v:v.quantile(.75),positive_folds=lambda v:int((v>0).sum()),negative_folds=lambda v:int((v<0).sum())).reset_index()
    write_csv(cs,out/'linear_coefficient_summary.csv')
    print(summary.groupby('model').head(5).to_string(index=False))
if __name__=='__main__':main()
