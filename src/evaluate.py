import sys
import yaml
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.preprocess import load_data, split_features_target


CONFIG_PATH = "configs/config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def evaluate_model():
    config = load_config()

    data_config = config["data"]
    training_config = config["training"]

    # Load dataset
    df = load_data(data_config["path"])

    X, y = split_features_target(
        df,
        data_config["target"],
    )

    # Recreate the exact same test split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=y,
    )

    # Load trained pipeline
    model = joblib.load(training_config["model_output"])

    predictions = model.predict(X_test)

    # Calculate metrics
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

    print("\nModel Evaluation")
    print("-----------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    minimum_accuracy = training_config["minimum_accuracy"]

    print(f"\nRequired minimum accuracy: {minimum_accuracy:.4f}")

    if accuracy < minimum_accuracy:
        print("FAILED: Model performance is below the required threshold.")
        sys.exit(1)

    print("PASSED: Model performance meets the required threshold.")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    evaluate_model()