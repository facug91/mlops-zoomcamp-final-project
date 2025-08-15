import argparse
import os
import sys

from mlflow.tracking import MlflowClient

import mlflow

MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000")


def move_model_to_production(model_name: str):
    mlflow.set_tracking_uri(MLFLOW_URL)
    client = MlflowClient()

    model_name = f"fruits-classifier-{model_name}"
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"No versions found for model '{model_name}'")
        sys.exit(1)

    best_version = None
    best_acc = float("-inf")
    for v in versions:
        run = client.get_run(v.run_id)
        acc = run.data.metrics.get("accuracy")
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_version = v
    if not best_version:
        print(f"No versions of '{model_name}' have 'accuracy' metric.")
        sys.exit(1)

    print(f"Moving model '{model_name}' version {best_version.version} to Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=best_version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Model '{model_name}' version {best_version.version} is now in Production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move a registered MLflow model to Production stage.")
    parser.add_argument(
        "--model-name",
        choices=["mobilenet_v3_small", "resnet50"],
        required=True,
        help="Name of the registered model in MLflow.",
    )

    args = parser.parse_args()
    move_model_to_production(args.model_name)
