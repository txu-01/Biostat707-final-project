import unittest
import numpy as np
from train_svm import make_model


class SVMCalibrationTests(unittest.TestCase):
    def test_scaler_is_inside_calibration_and_fitted_on_supplied_training_only(self):
        rng = np.random.RandomState(707)
        X = rng.normal(size=(36, 3))
        y = np.tile([0, 1], 18)
        held_out = np.full((4, 3), 10000.0)
        model = make_model('Linear_SVM', {'C': 0.1})
        self.assertEqual(model.estimator.steps[0][0], 'scale')
        self.assertEqual(model.cv.n_splits, 3)
        self.assertFalse(model.ensemble)
        model.fit(X, y)
        scaler = model.calibrated_classifiers_[0].estimator.named_steps['scale']
        before = scaler.mean_.copy()
        np.testing.assert_allclose(before, X.mean(axis=0))
        probabilities = model.predict_proba(held_out)
        np.testing.assert_array_equal(before, scaler.mean_)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)


if __name__ == '__main__':
    unittest.main()
