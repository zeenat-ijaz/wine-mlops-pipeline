"""Load the @champion model from the registry and score it on the test split."""
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_data
from src.train import MODEL_NAME, TRACKING_URI


def evaluate_champion():
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    _, X_test, _, y_test = load_data()
    preds = model.predict(X_test)
    results = {
        "test_accuracy": accuracy_score(y_test, preds),
        "test_f1_macro": f1_score(y_test, preds, average="macro"),
        "test_log_loss": log_loss(y_test, model.predict_proba(X_test)),
    }
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    return results


if __name__ == "__main__":
    evaluate_champion()
