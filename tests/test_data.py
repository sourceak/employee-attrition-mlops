import pandas as pd


DATA_PATH = "data/employee_attrition.csv"


def test_expected_columns_exist():
    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "Age",
        "Attrition",
        "Department",
        "JobRole",
        "MonthlyIncome",
        "YearsAtCompany",
    }

    assert required_columns.issubset(df.columns)


def test_target_contains_expected_values():
    df = pd.read_csv(DATA_PATH)

    target_values = set(df["Attrition"].dropna().unique())

    assert target_values == {"Yes", "No"}


def test_numeric_values_are_in_valid_ranges():
    df = pd.read_csv(DATA_PATH)

    assert df["Age"].dropna().between(18, 100).all()
    assert (df["MonthlyIncome"].dropna() > 0).all()
    assert (df["YearsAtCompany"].dropna() >= 0).all()