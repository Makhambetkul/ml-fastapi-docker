import json
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"
INFO_PATH = BASE_DIR / "model_info.json"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "iris-classification"
REGISTERED_MODEL_NAME = "iris-classifier"
MODEL_ALIAS = "champion"
RANDOM_STATE = 42

CANDIDATES = [
    ("logreg_C0.1", "LogisticRegression", {"C": 0.1, "max_iter": 1000}),
    ("logreg_C1", "LogisticRegression", {"C": 1.0, "max_iter": 1000}),
    ("logreg_C10", "LogisticRegression", {"C": 10.0, "max_iter": 1000}),
    ("rf_50_depth3", "RandomForest", {"n_estimators": 50, "max_depth": 3}),
    ("rf_200_full", "RandomForest", {"n_estimators": 200, "max_depth": None}),
]


def build_model(model_type, params):
    if model_type == "LogisticRegression":
        clf = LogisticRegression(random_state=RANDOM_STATE, **params)
    elif model_type == "RandomForest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, **params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def main():
    os.chdir(BASE_DIR)

    iris = load_iris()
    X = iris.data
    y = iris.target_names[iris.target]  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}\n")

    results = []
    for run_name, model_type, params in CANDIDATES:
        with mlflow.start_run(run_name=run_name) as run:
            model = build_model(model_type, params)

            cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "precision_macro": precision_score(y_test, y_pred, average="macro"),
                "recall_macro": recall_score(y_test, y_pred, average="macro"),
                "cv_f1_mean": cv_f1.mean(),
                "cv_f1_std": cv_f1.std(),
            }

            mlflow.log_param("model_type", model_type)
            mlflow.log_params(params)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", RANDOM_STATE)

            mlflow.log_metrics(metrics)

            with tempfile.TemporaryDirectory() as tmp:
                model_file = Path(tmp) / "model.joblib"
                joblib.dump(model, model_file)
                mlflow.log_artifact(str(model_file), artifact_path="model_file")

                report_file = Path(tmp) / "classification_report.txt"
                report_file.write_text(classification_report(y_test, y_pred))
                mlflow.log_artifact(str(report_file), artifact_path="reports")

            signature = infer_signature(X_train, model.predict(X_train))
            extra = {}
            if model_type == "RandomForest":
                extra["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train[:3],
                **extra,
            )

            mlflow.set_tag("dataset", "iris")
            results.append({
                "run_id": run.info.run_id,
                "run_name": run_name,
                "model_type": model_type,
                "params": params,
                "metrics": metrics,
                "model": model,
            })
            print(f"{run_name:<14} cv_f1={metrics['cv_f1_mean']:.3f}  "
                  f"test_acc={metrics['accuracy']:.3f}  test_f1={metrics['f1_macro']:.3f}")

    best = max(results, key=lambda r: (r["metrics"]["cv_f1_mean"], r["metrics"]["f1_macro"]))
    print(f"\nBest run: {best['run_name']} (run_id={best['run_id']})")

    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    version = registered.version

    client = MlflowClient()
    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description="Classifies iris flowers (setosa, versicolor, virginica) "
                    "from 4 flower measurements.",
    )
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=f"Best run: {best['run_name']} "
                    f"(cv_f1={best['metrics']['cv_f1_mean']:.3f}, "
                    f"test_acc={best['metrics']['accuracy']:.3f})",
    )
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "model_type", best["model_type"])
    print(f"Registered model: {REGISTERED_MODEL_NAME}, version {version}, alias '{MODEL_ALIAS}'")

    joblib.dump(best["model"], MODEL_PATH)
    info = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(version),
        "alias": MODEL_ALIAS,
        "run_id": best["run_id"],
        "run_name": best["run_name"],
        "model_type": best["model_type"],
        "params": best["params"],
        "metrics": {k: round(float(v), 4) for k, v in best["metrics"].items()},
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"Saved {MODEL_PATH.name} and {INFO_PATH.name}")


if __name__ == "__main__":
    main()
