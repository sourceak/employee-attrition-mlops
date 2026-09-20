# Data Drift Monitoring

## Overview

This project uses Evidently to monitor data drift between the reference training dataset and simulated production data.

The monitoring script evaluates all model features and generates an HTML drift report at:

`reports/drift_report.html`

The maximum allowed share of drifted features is configured in `configs/config.yaml`.

## Drift Results

The monitoring analysis evaluated 34 features.

- Total features: 34
- Drifted features: 3
- Overall drift share: 8.82%
- Configured drift threshold: 20%

The following features showed detected drift:

- Age
- MonthlyIncome
- JobRole

Because 8.82% of the features drifted, which is below the configured 20% threshold, the monitoring check passed.

## Why Did These Features Drift?

The production dataset is simulated intentionally to demonstrate how the monitoring system responds to changing data.

### Age

The simulated production data shifts employee ages upward. This changes the distribution of the Age feature compared with the original training data.

### MonthlyIncome

MonthlyIncome values are increased by 35% in the simulated production dataset. This creates a substantial change in the income distribution.

### JobRole

The simulated production data changes a portion of JobRole values to Research Scientist. This changes the distribution of job roles compared with the training dataset.

## Potential Model Impact

Feature drift means that the data the model receives in production may differ from the data used during training.

Because Age, MonthlyIncome, and JobRole are model inputs, persistent changes in these distributions could affect model predictions and reduce performance.

Drift alone does not prove that model performance has decreased. Performance should also be evaluated using labeled production data when it becomes available.

## Recommended Action

If similar drift appeared in a real production system, I would:

1. Investigate whether the drift represents a real change in the employee population or a data collection problem.
2. Verify that the production data pipeline is operating correctly.
3. Evaluate model performance on recent labeled production data.
4. Continue monitoring the drift to determine whether it is temporary or persistent.
5. Retrain and validate the model using newer representative data if the drift persists and negatively affects model performance.

## Running Drift Monitoring

Run:

`python -m src.monitor_drift`

The command prints the detected drifted features and overall drift share and creates the Evidently HTML report.

The process exits with a non-zero status if the share of drifted features exceeds the configured threshold.