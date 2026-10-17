import pytest
import pandas as pd
import numpy as np
from serverlex.preprocessing.clean import Clean


def test_clean_with_dataframe():
    df = pd.DataFrame({
        "First Name": [" Alice ", "Bob", "Charlie "],
        "Score": [10.0, 20.0, 1000.0]  # Outlier
    })
    cleaner = Clean(df)
    clean_df = cleaner.total_clean(outlier_method="IQR", outlier_param="m")

    # Verify snake_casing
    assert "first_name" in clean_df.columns
    assert "score" in clean_df.columns

    # Verify whitespace removal
    assert clean_df["first_name"].tolist() == ["alice", "bob", "charlie"]


def test_clean_fill_missing():
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["apple", np.nan, "apple", "banana"]
    })
    cleaner = Clean(df)
    clean_df = cleaner.total_clean(missing_param="m")
    assert clean_df["num"].isna().sum() == 0
    assert clean_df["cat"].isna().sum() == 0
