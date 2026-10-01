"""Hyperparameter search for RF and GBM with MLflow tracking and model registry."""
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import SEED, load_data

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"
CV_FOLDS = 5

SEARCH_GRID = {
    "RandomForest": [
        {"n_estimators": 60, "max_depth": 3, "min_samples_split": 2},
        {"n_estimators": 120, "max_depth": 6, "min_samples_split": 4},
        {"n_estimators": 250, "max_depth": None, "min_samples_split": 2},
    ],
    "GradientBoosting": [
        {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 2},
        {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 180, "learning_rate": 0.1, "max_depth": 3},
    ],
}

SCORING = {"f1_macro": "f1_macro", "accuracy": "accuracy", "log_loss": "neg_log_loss"}


def build_model(family, params):
    """Create an unfitted classifier for the given model family."""
    if family == "RandomForest":
        return RandomForestClassifier(random_state=SEED, **params)
    if family == "GradientBoosting":
        return GradientBoostingClassifier(random_state=SEED, **params)
    raise ValueError(f"Unknown model family: {family}")


def cross_validate_model(model, X, y):
    """Run 5-fold stratified CV and return mean train/validation metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING, return_train_score=True)
    metrics = {}
    for name in SCORING:
        train_val = res[f"train_{name}"].mean()
        val_val = res[f"test_{name}"].mean()
        if name == "log_loss":  # sklearn reports negative log loss
            train_val, val_val = -train_val, -val_val
        metrics[f"train_{name}"] = float(train_val)
        metrics[f"val_{name}"] = float(val_val)
    return metrics


def run_search():
    """Log one MLflow run per configuration; return the best run by val macro F1."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, _, y_train, _ = load_data()

    best_run_id, best_f1 = None, -1.0
    for family, configs in SEARCH_GRID.items():
        for i, params in enumerate(configs, start=1):
            with mlflow.start_run(run_name=f"{family}-config-{i}") as run:
                model = build_model(family, params)
                metrics = cross_validate_model(model, X_train, y_train)
                model.fit(X_train, y_train)

                mlflow.log_params(params)
                mlflow.log_param("cv_folds", CV_FOLDS)
                mlflow.log_metrics(metrics)
                mlflow.set_tags({"model_family": family, "seed": SEED})

                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train.head(5),
                )
                print(f"{run.info.run_name}: val_f1_macro={metrics['val_f1_macro']:.4f}")

                if metrics["val_f1_macro"] > best_f1:
                    best_f1, best_run_id = metrics["val_f1_macro"], run.info.run_id
    return best_run_id, best_f1


def register_champion(run_id):
    """Register the run's model as WineClassifier and alias it as champion."""
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version.version)
    print(f"Registered {MODEL_NAME} v{version.version} as @champion")


if __name__ == "__main__":
    run_id, f1 = run_search()
    print(f"Best run {run_id} with val_f1_macro={f1:.4f}")
    register_champion(run_id)
