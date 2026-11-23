import os
import unittest
import numpy as np
np.seterr(all='ignore')
import pandas as pd
from app import app, HoKashyapClassifier, find_elbow_point

class TestClassifiersApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    def test_ho_kashyap_separable_dataset(self):
        """Test Ho-Kashyap procedure on guaranteed linearly separable data."""
        np.random.seed(42)
        X0 = np.random.randn(25, 2) + np.array([-2.5, -2.5])
        X1 = np.random.randn(25, 2) + np.array([2.5, 2.5])
        X = np.vstack([X0, X1])
        y = np.array([0]*25 + [1]*25)

        model = HoKashyapClassifier(eta=0.5, max_iter=1000, eps=1e-3)
        model.fit(X, y)

        self.assertEqual(model.status_, 'separable')
        self.assertEqual(model.history_misc_[-1], 0)
        self.assertGreater(float(np.min(model.b_)), 0)
        self.assertEqual(len(model.a_), 3) # bias + 2 features

        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        self.assertEqual(accuracy, 1.0)

    def test_ho_kashyap_inseparable_dataset(self):
        """Test Ho-Kashyap procedure on guaranteed non-linearly separable data."""
        np.random.seed(42)
        # Concentric circles
        r0 = np.random.uniform(0.5, 1.0, 30)
        t0 = np.random.uniform(0, 2*np.pi, 30)
        X0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)

        r1 = np.random.uniform(2.5, 3.5, 30)
        t1 = np.random.uniform(0, 2*np.pi, 30)
        X1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)

        X = np.vstack([X0, X1])
        y = np.array([0]*30 + [1]*30)

        model = HoKashyapClassifier(eta=0.5, max_iter=200, eps=1e-4)
        model.fit(X, y)

        self.assertIn(model.status_, ['inseparable', 'max_iter'])
        self.assertGreater(model.history_misc_[-1], 0)

    def test_flask_routes_full_pipeline(self):
        """Test complete web workflow: upload -> train Ho-Kashyap -> predict -> reset."""
        # 1. Reset
        resp = self.client.get('/reset', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

        # 2. Upload Linearly Separable Benchmark
        resp = self.client.post('/', data={
            'action': 'upload_dataset',
            'sample_dataset': 'Linearly_Separable_2D.csv'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Linearly_Separable_2D.csv', resp.data)

        # 3. Train Ho-Kashyap Model
        resp = self.client.post('/', data={
            'action': 'train_model',
            'target_var': 'Target',
            'classifier_type': 'ho_kashyap',
            'use_standardization': '1',
            'default_eta': '0.5',
            'max_iter': '1000',
            'eps_val': '0.0001',
            'eta_values': ['0.1', '0.3', '0.5', '0.7', '0.9']
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Ho-Kashyap Linear Classifier Results', resp.data)
        self.assertIn(b'Final Weight Vector', resp.data)
        self.assertIn(b'Optimal Learning Rate', resp.data)

        # 4. Predict new instance
        resp = self.client.post('/', data={
            'action': 'predict_instance',
            'Feature_1': '2.0',
            'Feature_2': '2.0'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Prediction Outcome: Class 1', resp.data)

    def test_flask_comparative_benchmark(self):
        """Test comparative benchmark workflow training all three classifiers."""
        # Load heart benchmark
        self.client.post('/', data={
            'action': 'upload_dataset',
            'sample_dataset': 'heart.csv'
        }, follow_redirects=True)

        resp = self.client.post('/', data={
            'action': 'train_model',
            'target_var': 'target',
            'classifier_type': 'all',
            'use_standardization': '1',
            'default_eta': '0.5',
            'max_iter': '500',
            'eps_val': '0.0001',
            'eta_values': ['0.2', '0.5', '0.8']
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Side-by-Side Model Benchmark', resp.data)
        self.assertIn(b'Ho-Kashyap Linear Classifier', resp.data)
        self.assertIn(b'K-Nearest Neighbors', resp.data)
        self.assertIn(b'Naive Bayes', resp.data)


if __name__ == '__main__':
    unittest.main()
