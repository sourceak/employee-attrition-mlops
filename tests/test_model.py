import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor


DATA_PATH = "data/employee_attrition.csv"


def train_test_model():
    """Train a model specifically for model validation tests."""

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


def test_model_prediction_shape_and_values():
    model, X_test, _ = train_test_model()

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({"Yes", "No"})


def test_model_meets_minimum_accuracy():
    model, X_test, y_test = train_test_model()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.75