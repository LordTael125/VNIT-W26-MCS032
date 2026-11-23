import os
import sys

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app

def run_tests():
    print("Running Flask test client verification...")
    client = app.test_client()

    # 1. Test GET /
    res = client.get('/')
    assert res.status_code == 200, f"GET / failed with {res.status_code}"
    assert b"Classifiers Studio" in res.data, "Brand title not in page"
    print("[PASS] 1. GET / returned 200 and rendered index.html")

    # 2. Test Upload Sample Dataset (Iris_Data.csv)
    res = client.post('/', data={
        'action': 'upload_dataset',
        'sample_dataset': 'Iris_Data.csv'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Dataset Inspection" in res.data, "Dataset inspection section not rendered"
    assert b"Iris_Data.csv" in res.data
    print("[PASS] 2. POST upload_dataset loaded Iris_Data.csv with inspection summary")

    # 3. Test Train Model (KNN + Naive Bayes on Iris)
    res = client.post('/', data={
        'action': 'train_model',
        'target_var': 'species',
        'classifier_type': 'both',
        'use_standardization': '1',
        'max_k': '30'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Model Evaluation" in res.data
    assert b"Optimal Elbow Point" in res.data or b"k_vs_error_rate.png" in res.data
    assert b"Classifier Benchmark Comparison" in res.data
    print("[PASS] 3. POST train_model evaluated KNN (K=1 & Optimal Elbow) and Naive Bayes")

    # 4. Test Predict Instance
    res = client.post('/', data={
        'action': 'predict_instance',
        'sepal_length': '5.1',
        'sepal_width': '3.5',
        'petal_length': '1.4',
        'petal_width': '0.2'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Classifier Diagnostic Output" in res.data
    assert b"Iris-setosa" in res.data or b"setosa" in res.data
    print("[PASS] 4. POST predict_instance correctly classified test vector to Iris-setosa")

    # 5. Test Reset
    res = client.get('/reset', follow_redirects=True)
    assert res.status_code == 200
    assert b"Session reset successfully" in res.data
    print("[PASS] 5. GET /reset cleaned uploads and cached state successfully")

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")

if __name__ == '__main__':
    run_tests()
