import os

import pandas as pd


REFERENCE_DATA_PATH = "LAP.csv"
CURRENT_DATA_PATH = "logs/predictions.csv"
REPORT_OUTPUT_PATH = "monitoring/reports/evidently_drift_report.html"

MONITORING_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]


def main():
    print("Starting Evidently monitoring report generation...")

    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_DATA_PATH}")

    if not os.path.exists(CURRENT_DATA_PATH):
        raise FileNotFoundError(f"Prediction log not found: {CURRENT_DATA_PATH}")

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    current_df = pd.read_csv(CURRENT_DATA_PATH)

    print(f"Reference dataset: {REFERENCE_DATA_PATH}")
    print(f"Reference rows: {reference_df.shape[0]}")
    print(f"Reference columns: {reference_df.shape[1]}")

    print(f"Current prediction log: {CURRENT_DATA_PATH}")
    print(f"Current rows: {current_df.shape[0]}")
    print(f"Current columns: {current_df.shape[1]}")

    available_columns = [
        col for col in MONITORING_COLUMNS
        if col in reference_df.columns and col in current_df.columns
    ]

    if not available_columns:
        raise ValueError("No matching monitoring columns found in reference and current data.")

    print("Columns included in Evidently report:")
    for col in available_columns:
        print(f"- {col}")

    reference_monitoring_df = reference_df[available_columns].copy()
    current_monitoring_df = current_df[available_columns].copy()

    for col in available_columns:
        reference_monitoring_df[col] = pd.to_numeric(
            reference_monitoring_df[col],
            errors="coerce"
        )
        current_monitoring_df[col] = pd.to_numeric(
            current_monitoring_df[col],
            errors="coerce"
        )

    reference_monitoring_df = reference_monitoring_df.dropna()
    current_monitoring_df = current_monitoring_df.dropna()

    print(f"Reference rows after cleaning: {reference_monitoring_df.shape[0]}")
    print(f"Current rows after cleaning: {current_monitoring_df.shape[0]}")

    if current_monitoring_df.empty:
        raise ValueError("Current monitoring data is empty after cleaning.")

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)

    # Evidently is used here only for report generation.
    # It is not part of the Flask prediction container.
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])

    report.run(
        reference_data=reference_monitoring_df,
        current_data=current_monitoring_df
    )

    report.save_html(REPORT_OUTPUT_PATH)

    print("Evidently report created successfully.")
    print(f"Report path: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()