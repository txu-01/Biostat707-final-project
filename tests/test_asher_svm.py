import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import final_analysis as common
from asher.svm import make_svm,run_shared_svm,validate_predictions,expected_keys

class AsherSVMTests(unittest.TestCase):
    def test_common_entry_routes_both_svm_models_to_owner(self):
        data=common.load_data()
        with tempfile.TemporaryDirectory() as tmp,patch('asher.svm.run_shared_svm') as route:
            common.run_primary_nested(data,Path(tmp),['Linear_SVM','RBF_SVM'],1,5,5)
            self.assertEqual(route.call_count,2)
            self.assertEqual([c.args[2] for c in route.call_args_list],[['Linear_SVM'],['RBF_SVM']])

    def test_scaler_never_sees_external_validation_rows(self):
        from sklearn.preprocessing import StandardScaler
        data=common.load_data();x=data[common.FEATURE_NAMES].iloc[:120];y=data.target.iloc[:120]
        original=StandardScaler.fit;seen=[]
        def record(s,values,*a,**kw):
            seen.append(set(values.index));return original(s,values,*a,**kw)
        with patch.object(StandardScaler,'fit',record):
            model=make_svm('Linear_SVM',{'C':.1}).fit(x.iloc[:100],y.iloc[:100])
            model.predict_proba(x.iloc[100:])
        self.assertEqual(len(seen),4)
        self.assertTrue(all(idx.issubset(set(x.index[:100])) for idx in seen))
        self.assertEqual(sum(len(idx)==100 for idx in seen),1)

    def test_archived_results_cannot_be_silently_overwritten(self):
        data=common.load_data()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'predictions/Linear_SVM.csv';p.parent.mkdir();p.write_text('archived bytes')
            with self.assertRaisesRegex(ValueError,'Refusing to overwrite'):
                run_shared_svm(data,Path(tmp),['Linear_SVM'],1)
            self.assertEqual(p.read_text(),'archived bytes')

    def test_bad_fold_or_label_is_rejected(self):
        data=common.load_data();p=expected_keys(data,1).assign(model='Linear_SVM',y_prob=.4,threshold=.5)
        validate_predictions(p,data,'Linear_SVM',1)
        for field in ['fold','y_true']:
            bad=p.copy();bad.loc[0,field]=1-int(bad.loc[0,field])
            with self.assertRaises(AssertionError):validate_predictions(bad,data,'Linear_SVM',1)

if __name__=='__main__':unittest.main()
