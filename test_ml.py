import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def processed_data():
    """
    Load a sample of the census data and process it for training.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "census.csv")
    data = pd.read_csv(data_path).sample(n=1000, random_state=42)
    X, y, _, _ = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X, y


def test_train_model_uses_random_forest(processed_data):
    """
    Test that train_model returns a fitted RandomForestClassifier.
    """
    X, y = processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")


def test_inference_returns_binary_predictions(processed_data):
    """
    Test that inference returns a numpy array with one 0 or 1 prediction per row.
    """
    X, y = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_expected_values():
    """
    Test that compute_model_metrics returns the expected precision, recall and F1.
    """
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)
