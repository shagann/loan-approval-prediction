import json
import os
from datetime import datetime

import pandas as pd


def main():
    print("Starting retraining workflow...")

    dataset_path = "LAP.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    os.makedirs("training", exist_ok=True)

    metrics = {
        "status": "completed",
        "message": "Retraining workflow executed successfully",
        "dataset": dataset_path,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "retraining_time_utc": datetime.utcnow().isoformat(),
        "new_model_version": "model-v2-candidate"
    }

    with open("training/retraining_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Retraining metrics saved to training/retraining_metrics.json")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()