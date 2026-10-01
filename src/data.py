"""Data loading, validation and splitting for the Wine dataset."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SEED = 42
N_FEATURES = 13
TEST_SIZE = 0.2


def validate_data(X, y):
    """Raise ValueError if the dataset has nulls or the wrong feature count."""
    if X.isnull().values.any() or y.isnull().any():
        raise ValueError("Dataset contains null values")
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {X.shape[1]}")


def load_data():
    """Load the Wine data and return a stratified 80/20 train-test split."""
    wine = load_wine(as_frame=True)
    X, y = wine.data, wine.target
    validate_data(X, y)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
