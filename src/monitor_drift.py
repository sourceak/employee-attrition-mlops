import os
import sys
import yaml
import numpy as np
import pandas as pd

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


CONFIG_PATH = "configs/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def simulate_production_data(reference_data):
    """Create simulated production data with intentional drift."""

    production_data = reference_data.copy()

    rng = np.random.default_rng(42)

    # Simulate numeric drift
    production_data["MonthlyIncome"] = (
        production_data["MonthlyIncome"] * 1.35
    )

    production_data["Age"] = (
        production_data["Age"]
        + rng.normal(5, 2, len(production_data))
    )

    # Simulate categorical drift
    valid_indices = production_data["JobRole"].dropna().index

    if len(valid_indices) > 0:
        selected = rng.choice(
            valid_indices,
            size=int(len(valid_indices) * 0.30),
            replace=False,
        )

        production_data.loc[
            selected, "JobRole"
        ] = "Research Scientist"

    return production_data


def monitor_drift():
    config = load_config()

    data_path = config["data"]["path"]
    target = config["data"]["target"]

    # Read the allowed drift threshold from config
    drift_threshold = config["monitoring"]["drift_threshold"]

    # Load training/reference data
    df = pd.read_csv(data_path)

    # Remove target because we are monitoring feature drift
    reference_data = df.drop(columns=[target]).copy()

    # Create simulated production data
    production_data = simulate_production_data(reference_data)

    # Create Evidently drift report
    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    report.run(
        reference_data=reference_data,
        current_data=production_data,
    )

    # Save HTML report
    os.makedirs("reports", exist_ok=True)

    report_path = "reports/drift_report.html"
    report.save_html(report_path)

    # Get report results
    results = report.as_dict()

    # Metric 0 contains overall dataset drift information
    overall_result = results["metrics"][0]["result"]

    drift_share = overall_result["share_of_drifted_columns"]
    number_drifted = overall_result["number_of_drifted_columns"]

    # Metric 1 contains drift information for every feature
    drift_table_result = results["metrics"][1]["result"]

    drift_by_columns = drift_table_result["drift_by_columns"]

    # Find the names of features where drift was detected
    drifted_features = []

    for feature, information in drift_by_columns.items():
        if information["drift_detected"]:
            drifted_features.append(feature)

    # Print results
    print("\nData Drift Monitoring")
    print("--------------------------------")
    print(f"Total features: {len(reference_data.columns)}")
    print(f"Drifted features: {number_drifted}")
    print(f"Overall drift share: {drift_share:.2%}")

    print("\nFeatures with detected drift:")

    for feature in drifted_features:
        print(f"- {feature}")

    print(f"\nHTML report saved to: {report_path}")
    print(f"Allowed drift threshold: {drift_threshold:.2%}")

    # Fail when drift exceeds configured threshold
    if drift_share > drift_threshold:
        print("\nFAILED: Drift exceeded the configured threshold.")
        sys.exit(1)

    print("\nPASSED: Drift is within the configured threshold.")


if __name__ == "__main__":
    monitor_drift()