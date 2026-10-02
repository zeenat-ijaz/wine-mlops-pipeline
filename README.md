# Wine MLOps Pipeline

![CI](https://github.com/zeenat-ijaz/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

Reproducible MLOps pipeline for classifying wine cultivars (`sklearn.datasets.load_wine`,
178 samples, 13 features, 3 classes). Two tree-based model families (Random Forest and
Gradient Boosting) are tuned with 5-fold stratified cross-validation and tracked in MLflow.
The best model is registered as `WineClassifier@champion`. A GitHub Actions quality gate
blocks models that fall below the required standard.

## Project structure

```
src/data.py        load, validate, stratified 80/20 split (seed 42)
src/train.py       hyperparameter search, MLflow logging, champion registration
src/evaluate.py    loads WineClassifier@champion and scores the test split
tests/             data tests and the model quality gate
Makefile           install / lint / test / train / evaluate / clean
```

## Usage

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install
make lint
make test
make train                       # 6 MLflow runs + registers the champion
make evaluate                    # test-set metrics of the champion
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Quality gate (`tests/test_model_gate.py`)

| Check | Rule |
|---|---|
| Metric threshold | validation macro F1 >= 0.88 |
| Inference latency | batch prediction <= 30 ms |
| Output schema | predictions only in {0, 1, 2} |
