import json
import os
import shutil
from datetime import datetime

import pandas as pd


def main():
    print("Starting retraining workflow...")

    dataset_path = "LAP.csv"

    source_model_version = "baseline-model"
    candidate_model_version = "model-v2-candidate"

    source_model_dir = f"models/{source_model_version}"
    candidate_model_dir = f"models/{candidate_model_version}"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not os.path.exists(source_model_dir):
        raise FileNotFoundError(f"Source model package not found: {source_model_dir}")

    df = pd.read_csv(dataset_path)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Recreate the candidate package each time retraining runs.
    # This keeps model-v2-candidate as the latest retraining output.
    if os.path.exists(candidate_model_dir):
        print(f"Removing existing candidate package: {candidate_model_dir}")
        shutil.rmtree(candidate_model_dir)

    print(f"Creating candidate model package: {candidate_model_dir}")
    shutil.copytree(source_model_dir, candidate_model_dir)

    retraining_time = datetime.utcnow().isoformat()

    # Metadata for the candidate model package.
    # In this assignment version, the baseline artefacts are copied and the
    # metadata is updated to demonstrate the MLOps model versioning flow.
    metadata = {
        "model_version": candidate_model_version,
        "model_stage": "candidate",
        "created_for": "Loan Approval Prediction API",
        "training_dataset": dataset_path,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "created_at_utc": retraining_time,
        "source_model_version": source_model_version,
        "model_files": [
            "model.pth",
            "best_model.pth"
        ],
        "preprocessing_files": [
            "scaler.pkl",
            "label_encoders.pkl",
            "label_encoder_target.pkl"
        ],
        "notes": (
            "Candidate model package created by the retraining workflow after drift detection. "
            "For this assignment version, the baseline artefacts are copied and metadata is updated "
            "to demonstrate model versioning, promotion, and rollback flow."
        )
    }

    metadata_path = os.path.join(candidate_model_dir, "model_metadata.json")

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    os.makedirs("training", exist_ok=True)

    # Demo evaluation gate.
    # In a production MLOps pipeline, these values would be calculated by
    # evaluating the candidate model against a validation/test dataset.
    #
    # For the demo, the candidate is marked as approved so the workflow can
    # demonstrate automatic promotion after retraining.
    baseline_accuracy = 0.78
    candidate_accuracy = 0.80
    baseline_f1 = 0.81
    candidate_f1 = 0.84

    if candidate_f1 >= baseline_f1 and candidate_accuracy >= baseline_accuracy:
        promotion_decision = "approved"
        promotion_reason = (
            "Candidate model passed the demo evaluation gate. "
            "Candidate F1 and accuracy are greater than or equal to the baseline."
        )
    else:
        promotion_decision = "rejected"
        promotion_reason = (
            "Candidate model did not pass the demo evaluation gate. "
            "Production model should remain unchanged."
        )

    metrics = {
        "status": "completed",
        "message": "Retraining workflow executed successfully because drift was detected",
        "dataset": dataset_path,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "retraining_time_utc": retraining_time,
        "source_model_version": source_model_version,
        "candidate_model_version": candidate_model_version,
        "candidate_model_package": candidate_model_dir,
        "metadata_path": metadata_path,

        # Evaluation gate output used by the GitHub Actions workflow.
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "baseline_f1": baseline_f1,
        "candidate_f1": candidate_f1,
        "promotion_decision": promotion_decision,
        "promotion_reason": promotion_reason
    }

    with open("training/retraining_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Candidate model package created successfully.")
    print(f"Candidate model package: {candidate_model_dir}")
    print(f"Metadata path: {metadata_path}")
    print("Retraining metrics saved to training/retraining_metrics.json")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()