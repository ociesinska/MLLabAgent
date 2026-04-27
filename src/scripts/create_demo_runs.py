from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow


def create_demo_runs(experiment_name: str, tracking_uri: str | None = None) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    demo_runs = [
        {
            "run_name": "baseline_lr",
            "params": {
                "model_type": "logistic_regression",
                "learning_rate": 0.01,
                "batch_size": 32,
                "augmentation": "none",
            },
            "metrics": {
                "accuracy": 0.81,
                "f1_score": 0.78,
                "precision": 0.79,
                "recall": 0.77,
            },
            "tags": {
                "project": "demo-ml-lab-agent",
                "stage": "baseline",
                "dataset": "demo-classification-v1",
            },
        },
        {
            "run_name": "augmented_cnn",
            "params": {
                "model_type": "simple_cnn",
                "learning_rate": 0.001,
                "batch_size": 32,
                "augmentation": "light",
            },
            "metrics": {
                "accuracy": 0.85,
                "f1_score": 0.82,
                "precision": 0.84,
                "recall": 0.81,
            },
            "tags": {
                "project": "demo-ml-lab-agent",
                "stage": "candidate",
                "dataset": "demo-classification-v1",
            },
        },
        {
            "run_name": "tuned_cnn",
            "params": {
                "model_type": "simple_cnn",
                "learning_rate": 0.0005,
                "batch_size": 64,
                "augmentation": "medium",
            },
            "metrics": {
                "accuracy": 0.89,
                "f1_score": 0.87,
                "precision": 0.88,
                "recall": 0.86,
            },
            "tags": {
                "project": "demo-ml-lab-agent",
                "stage": "best-so-far",
                "dataset": "demo-classification-v1",
            },
        },
    ]

    artifacts_dir = Path("artifacts/demo_runs")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for run in demo_runs:
        with mlflow.start_run(run_name=run["run_name"]):
            mlflow.log_params(run["params"])
            mlflow.log_metrics(run["metrics"])
            mlflow.set_tags(run["tags"])

            summary = {
                "run_name": run["run_name"],
                "notes": f"Demo run created for experiment analysis assistant. "
                f"Best metric snapshot: accuracy={run['metrics']['accuracy']}, "
                f"f1_score={run['metrics']['f1_score']}.",
                "params": run["params"],
                "metrics": run["metrics"],
            }

            artifact_path = artifacts_dir / f"{run['run_name']}_summary.json"
            artifact_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(artifact_path), artifact_path="run_summary")

            print(f"Created run: {run['run_name']} | accuracy={run['metrics']['accuracy']} | f1_score={run['metrics']['f1_score']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create demo MLflow runs.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="MLLabAgent Demo Runs",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="Optional MLflow tracking URI. Falls back to MLFLOW_TRACKING_URI env var.",
    )
    args = parser.parse_args()

    create_demo_runs(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
