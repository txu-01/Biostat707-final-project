"""Train both SVMs from scratch, summarize, and optionally compare archives."""
import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
_handles=[]
if os.name=='nt' and (Path(sys.prefix)/'Library/bin').is_dir():
    folder=str(Path(sys.prefix)/'Library/bin');_handles.append(os.add_dll_directory(folder));os.environ['PATH']=folder+os.pathsep+os.environ.get('PATH','')
sys.path.insert(0,str(ROOT/'src'))
import argparse,json
from final_analysis import load_data,summarize_primary
from asher.svm import run_shared_svm,compare_archive
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/asher/svm_reproduction')
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--models',nargs='+',choices=['Linear_SVM','RBF_SVM'],default=['Linear_SVM','RBF_SVM'])
    p.add_argument('--verify-against',type=Path)
    a=p.parse_args();run_shared_svm(load_data(),a.output,a.models,a.repeats)
    summarize_primary(a.output)
    if a.verify_against:
        checks=compare_archive(a.output,a.verify_against,a.models);print(json.dumps(checks,indent=2))
        if not all(c['matches_1e_7'] for c in checks):raise SystemExit('Refit differs from archive; keep results separate and investigate')
if __name__=='__main__':main()
