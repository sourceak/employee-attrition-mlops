import mlflow
import yaml


CONFIG_PATH = "configs/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def compare_experiments():
    config = load_config()

    experiment_name = config["training"]["experiment_name"]
    primary_metric = config["training"]["primary_metric"]

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{experiment_name}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    if runs.empty:
        raise ValueError("No MLflow runs were found.")

    metric_column = f"metrics.{primary_metric}"

    successful_runs = runs.dropna(subset=[metric_column])

    if successful_runs.empty:
        raise ValueError(
            f"No runs contain the primary metric '{primary_metric}'."
        )

    best_index = successful_runs[metric_column].idxmax()
    best_run = successful_runs.loc[best_index]

    print("\nMLflow Experiment Comparison")
    print("--------------------------------")
    print(f"Experiment: {experiment_name}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Primary metric: {primary_metric}")

    print("\nBest Run")
    print("--------------------------------")
    print(f"Run ID: {best_run['run_id']}")
    print(f"{primary_metric}: {best_run[metric_column]:.4f}")

    print("\nHyperparameters")
    print("--------------------------------")

    parameter_columns = [
        column
        for column in successful_runs.columns
        if column.startswith("params.")
    ]

    for column in parameter_columns:
        parameter_name = column.replace("params.", "")
        print(f"{parameter_name}: {best_run[column]}")


if __name__ == "__main__":
    compare_experiments()