import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset."""

    if not isinstance(path, str):
        raise TypeError("path must be a string")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("dataset is empty")

    return df


def split_features_target(df: pd.DataFrame, target_column: str):
    """Separate the features from the target without modifying the input."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipelines for numeric and categorical features."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if X.empty:
        raise ValueError("X cannot be empty")

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return preprocessor