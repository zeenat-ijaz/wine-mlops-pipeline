import numpy as np
import pytest

from src.data import N_FEATURES, load_data, validate_data


def test_split_shapes():
    X_train, X_test, _, _ = load_data()
    assert len(X_train) == 142 and len(X_test) == 36
    assert X_train.shape[1] == N_FEATURES


def test_split_is_stratified():
    _, _, y_train, y_test = load_data()
    train_ratio = y_train.value_counts(normalize=True).sort_index()
    test_ratio = y_test.value_counts(normalize=True).sort_index()
    assert np.allclose(train_ratio, test_ratio, atol=0.05)


def test_split_is_reproducible():
    first = load_data()[0]
    second = load_data()[0]
    assert first.index.equals(second.index)


def test_validation_rejects_nulls():
    X_train, _, y_train, _ = load_data()
    X_bad = X_train.copy()
    X_bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_data(X_bad, y_train)


def test_validation_rejects_wrong_feature_count():
    X_train, _, y_train, _ = load_data()
    with pytest.raises(ValueError):
        validate_data(X_train.iloc[:, :12], y_train)
