import os
import unittest
import numpy as np
np.seterr(all='ignore')
import pandas as pd
from app import app, ParzenWindowClassifier, HoKashyapClassifier, find_elbow_point

class TestParzenWindowClassifiersApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    def test_parzen_gaussian_logsumexp_stability(self):
        """Test Gaussian window with log-sum-exp trick under extreme underflow conditions (small h, high d)."""
        np.random.seed(42)
        X = np.random.randn(50, 10) * 5.0  # high variance, 10 features
        y = np.array([0] * 25 + [1] * 25)

        # Extreme small h = 0.001 (standard exp(-d^2 / 2h^2) would produce 0.0 underflow)
        model = ParzenWindowClassifier(h=0.001, window='gaussian')
        model.fit(X, y)

        dens, post, log_dens, empty_cnt, _ = model.predict_densities_and_posteriors(X[:5])
        
        # Verify no NaN or Inf in posteriors
        self.assertFalse(np.isnan(post).any())
        self.assertFalse(np.isinf(post).any())
        # Verify posteriors sum to 1.0 for each test sample
        np.testing.assert_allclose(np.sum(post, axis=1), np.ones(5), atol=1e-5)
        # Empty count for Gaussian is always 0 (infinite support)
        self.assertEqual(empty_cnt, 0)

    def test_parzen_hypercube_empty_windows(self):
        """Test Hypercube window empty-window detection and prior fallback."""
        np.random.seed(42)
        X_train = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
        y_train = np.array([0, 0, 1, 1])

        # Test point far away from all training points
        X_test = np.array([[5.0, 5.0], [0.05, 0.05]])
        
        # With h = 0.5:
        # Sample 1 ([5, 5]) falls in NO hypercube -> empty window
        # Sample 2 ([0.05, 0.05]) falls in Class 0's hypercube
        model = ParzenWindowClassifier(h=0.5, window='hypercube')
        model.fit(X_train, y_train)

        dens, post, _, empty_cnt, empty_mask = model.predict_densities_and_posteriors(X_test)
        self.assertEqual(empty_cnt, 1)
        self.assertTrue(empty_mask[0])
        self.assertFalse(empty_mask[1])
        # Sample 0 should fall back to prior (50% each)
        np.testing.assert_allclose(post[0], [0.5, 0.5])
        # Sample 1 should have non-zero density for class 0
        self.assertGreater(dens[1, 0], 0.0)

    def test_parzen_classification_iris(self):
        """Test Parzen Window classification on Iris benchmark."""
        sample_path = os.path.join(os.path.dirname(__file__), 'sample_datasets', 'Iris_Data.csv')
        df = pd.read_csv(sample_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        model = ParzenWindowClassifier(h=0.5, window='gaussian')
        model.fit(X, y)
        preds = model.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.90)

    def test_flask_routes_full_pipeline(self):
        """Test complete web workflow: upload -> train Parzen Window -> predict -> reset."""
        # 1. Reset
        resp = self.client.get('/reset', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

        # 2. Upload Iris Benchmark
        resp = self.client.post('/', data={
            'action': 'upload_dataset',
            'sample_dataset': 'Iris_Data.csv'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Iris_Data.csv', resp.data)
        self.assertIn(b'Data Inspection', resp.data)

        # 3. Train Parzen Window Model
        resp = self.client.post('/', data={
            'action': 'train_model',
            'target_var': 'species',
            'classifier_type': 'parzen_window',
            'use_standardization': '1',
            'h_values': '0.05, 0.1, 0.5, 1.0, 2.0'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Parzen Window Density Estimation', resp.data)
        self.assertIn(b'Default Parameters Run', resp.data)
        self.assertIn(b'Window Width ($h$) Analysis', resp.data)
        self.assertIn(b'Optimal Window Function', resp.data)
        self.assertIn(b'Test Set Performance Comparison', resp.data)

        # 4. Predict new instance
        resp = self.client.post('/', data={
            'action': 'predict_instance',
            'sepal_length': '5.1',
            'sepal_width': '3.5',
            'petal_length': '1.4',
            'petal_width': '0.2'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Prediction Outcome: Class', resp.data)
        self.assertIn(b'Posterior Probability Distribution', resp.data)
        self.assertIn(b'Estimated Class-Conditional Densities', resp.data)

    def test_flask_comparative_benchmark(self):
        """Test comparative benchmark workflow training all classifiers side-by-side."""
        # Upload heart benchmark
        self.client.post('/', data={
            'action': 'upload_dataset',
            'sample_dataset': 'heart.csv'
        }, follow_redirects=True)

        resp = self.client.post('/', data={
            'action': 'train_model',
            'target_var': 'target',
            'classifier_type': 'all',
            'use_standardization': '1'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Comprehensive Classifier Benchmark', resp.data)
        self.assertIn(b'benchmark_comparison.png', resp.data)


if __name__ == '__main__':
    unittest.main()
