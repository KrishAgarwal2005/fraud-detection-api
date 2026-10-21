"""Shared test setup.

The tests never need the real 150 MB dataset or the real trained model.
Instead we train a tiny logistic regression on random numbers, save it in a
temporary folder, and start the API with that model and a temporary database.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from fraud.api import create_app
from fraud.features import FEATURES


@pytest.fixture
def client(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    y = (X["V1"] > 1).astype(int)          # a simple rule the tiny model can learn
    model = LogisticRegression().fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "threshold": 0.5, "features": FEATURES,
                 "model_name": "test-model", "version": "test"}, model_path)
    app = create_app(str(model_path), str(tmp_path / "test.db"))
    return TestClient(app)


def make_transaction(v1: float = 0.0, amount: float = 50.0) -> dict:
    """A valid request body. Only V1 changes, which drives the test model."""
    return {"time": 3600 * 2, "amount": amount, "v": [v1] + [0.0] * 27}
