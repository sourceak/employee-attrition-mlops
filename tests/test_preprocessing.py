import pandas as pd
import numpy as np
import pytest

from src.preprocess import split_features_target, build_preprocessor


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Age": [25, 30, np.nan, 40],
        "Income": [50000, np.nan, 70000, 80000],
        "Department": ["Sales", "HR", np.nan, "Engineering"],
        "Attrition": ["No", "Yes", "No", "Yes"],
    })


def test_split_features_target(sample_data):
    X, y = split_features_target(sample_data, "Attrition")

    assert "Attrition" not in X.columns
    assert len(y) == 4


def test_split_does_not_modify_original(sample_data):
    original = sample_data.copy(deep=True)

    split_features_target(sample_data, "Attrition")

    pd.testing.assert_frame_equal(sample_data, original)


def test_missing_numeric_values_are_imputed(sample_data):
    X, _ = split_features_target(sample_data, "Attrition")
    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert not np.isnan(transformed).any()


def test_categorical_columns_are_encoded(sample_data):
    X, _ = split_features_target(sample_data, "Attrition")
    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[1] > 0
    assert np.issubdtype(transformed.dtype, np.number)


def test_invalid_dataframe_type_raises_error():
    with pytest.raises(TypeError):
        split_features_target(
            ["not", "a", "dataframe"],
            "Attrition",
        )


def test_missing_target_raises_error(sample_data):
    with pytest.raises(ValueError):
        split_features_target(
            sample_data,
            "DoesNotExist",
        )


def test_empty_features_raise_error():
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        build_preprocessor(empty_df)