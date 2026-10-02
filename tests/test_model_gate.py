"""MLOps quality gate: blocks models that are inaccurate, slow, or emit invalid labels.

The model is trained here (not loaded from the registry) so the gate also runs on CI,
where mlruns/ and mlflow.db are not available.
"""
import time

import pytest

from src.data import load_data
from src.train import SEARCH_GRID, build_model, cross_validate_model

F1_THRESHOLD = 0.90
LATENCY_MS = 30
VALID_CLASSES = {0, 1, 2}


@pytest.fixture(scope="module")
def trained():
    X_train, X_test, y_train, _ = load_data()
    model = build_model("RandomForest", SEARCH_GRID["RandomForest"][1])
    metrics = cross_validate_model(model, X_train, y_train)
    model.fit(X_train, y_train)
    return model, metrics, X_test


def test_metric_threshold_gate(trained):
    _, metrics, _ = trained
    assert metrics["val_f1_macro"] >= F1_THRESHOLD


def test_inference_latency_gate(trained):
    model, _, X_test = trained
    model.predict(X_test)  # warm-up
    start = time.perf_counter()
    model.predict(X_test)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= LATENCY_MS, f"Batch inference took {elapsed_ms:.1f} ms"


def test_output_schema_integrity(trained):
    model, _, X_test = trained
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert {int(p) for p in preds}.issubset(VALID_CLASSES)
