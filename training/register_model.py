"""Log the current candidate (calibration + training report) to MLflow and register it as `laya-router`.

Usage: python training/register_model.py [--tracking-uri sqlite:///mlflow.db]
Prints the new model version. The weights stay in models/ (too large for the tracking store); the registered
artifact is the small bundle that identifies the candidate: calibration, training report, thresholds.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ("models/calibration.json", "models/laya_ft/train_report.json", "config/policy.yaml",
          "docs/results/thresholds.json")


def main() -> None:
    import mlflow
    import mlflow.exceptions
    from promote import MODEL_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("laya-router")
    with mlflow.start_run() as run:
        for relative in BUNDLE:
            source = ROOT / relative
            if source.exists():
                mlflow.log_artifact(str(source), artifact_path="candidate")
        mlflow.log_param("checkpoint", "models/laya_ft")
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:  # already registered
        pass
    version = client.create_model_version(MODEL_NAME, f"{run.info.artifact_uri}/candidate", run.info.run_id)
    print(version.version)


if __name__ == "__main__":
    main()
