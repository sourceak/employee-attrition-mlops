import pandas as pd
import numpy as np

DATA_PATH = "data/employee_attrition.csv"

df = pd.read_csv(DATA_PATH)

# Reproducible random generator
rng = np.random.default_rng(42)

# Introduce missing values into a mix of numeric and categorical columns
columns = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobRole",
    "Department"
]

for column in columns:
    indices = rng.choice(
        df.index,
        size=int(len(df) * 0.02),
        replace=False
    )
    df.loc[indices, column] = np.nan

df.to_csv(DATA_PATH, index=False)

print("Missing values introduced successfully.")
print(df[columns].isnull().sum())
print("Total missing values:", df.isnull().sum().sum())