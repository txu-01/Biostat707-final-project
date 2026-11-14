"""Portable entry point; sets Windows conda DLL paths before scientific imports."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
DLL_HANDLES = []
if os.name == 'nt':
    candidates = [Path(sys.prefix) / 'Library' / 'bin', Path(sys.prefix) / 'DLLs']
    for candidate in candidates:
        if candidate.exists():
            DLL_HANDLES.append(os.add_dll_directory(str(candidate)))
            os.environ['PATH'] = str(candidate) + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
if len(sys.argv) != 2 or sys.argv[1] not in ['check', 'develop', 'finalize', 'verify', 'evaluate']:
    raise SystemExit('Usage: python scripts/run_svm.py {check|develop|finalize|verify|evaluate}')
if sys.argv[1] == 'check':
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
elif sys.argv[1] == 'evaluate':
    from evaluate_all import run_evaluation
    run_evaluation()
else:
    runpy.run_path(str(ROOT / 'src' / 'train_svm.py'), run_name='__main__')
