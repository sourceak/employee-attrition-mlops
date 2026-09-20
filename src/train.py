import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import load_data, split_features_target, build_preprocessor


CONFIG_PATH = "configs/config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def get_dvc_hash(dvc_file="data/employee_attrition.csv.dvc"):
    """Read the dataset version/hash stored by DVC."""

    with open(dvc_file, "r") as file:
        dvc_info = yaml.safe_load(file)

    return dvc_info["outs"][0]["md5"]


def train_model(config_path=None):
    config = load_config(config_path or CONFIG_PATH)

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    # -------------------------
    # Load data
    # -------------------------
    df = load_data(data_config["path"])

    X, y = split_features_target(
        df,
        data_config["target"]
    )

    # -------------------------
    # Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=y,
    )

    # -------------------------
    # Preprocessing
    # -------------------------
    preprocessor = build_preprocessor(X_train)

    # -------------------------
    # Model
    # -------------------------
    model = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        min_samples_split=model_config["min_samples_split"],
        min_samples_leaf=model_config["min_samples_leaf"],
        random_state=model_config["random_state"],
    )

    # Combine preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # -------------------------
    # MLflow
    # -------------------------
    mlflow.set_experiment(training_config["experiment_name"])

    dvc_hash = get_dvc_hash()

    with mlflow.start_run():

        # Log data/config information
        mlflow.log_param("data_version", dvc_hash)
        mlflow.log_param("target", data_config["target"])
        mlflow.log_param("test_size", data_config["test_size"])
        mlflow.log_param(
            "split_random_state",
            data_config["random_state"]
        )

        # Log ALL model hyperparameters from config
        for parameter, value in model_config.items():
            mlflow.log_param(parameter, value)

        # -------------------------
        # Train
        # -------------------------
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        # -------------------------
        # Metrics
        # -------------------------
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            pos_label="Yes",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label="Yes",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label="Yes",
            zero_division=0,
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log trained model as an MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        # -------------------------
        # Save local model
        # -------------------------
        model_path = training_config["model_output"]

        os.makedirs(
            os.path.dirname(model_path),
            exist_ok=True
        )

        joblib.dump(pipeline, model_path)

        # -------------------------
        # Results
        # -------------------------
        print("\nTraining complete")
        print("-----------------------------")
        print(f"DVC data version: {dvc_hash}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Model saved to: {model_path}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    CONFIG_PATH = args.config

    train_model()