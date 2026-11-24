import pytest
import numpy as np
import pandas as pd
from serverlex import Clean, MLPipeline

def test_clean_optimize_memory_integers():
    N = 1000
    df = pd.DataFrame({
        "tiny_int": np.random.randint(-100, 100, N).astype(np.int64),
        "med_int": np.random.randint(-30000, 30000, N).astype(np.int64),
        "large_int": np.random.randint(-1000000, 1000000, N).astype(np.int64),
    })
    cleaner = Clean(df)
    opt_df = cleaner.optimize_memory(verbose=False)

    assert opt_df["tiny_int"].dtype == np.int8
    assert opt_df["med_int"].dtype == np.int16
    assert opt_df["large_int"].dtype == np.int32
    assert cleaner.memory_summary["reduction_pct"] > 50.0


def test_clean_optimize_memory_floats():
    N = 1000
    df = pd.DataFrame({
        "val_a": np.random.randn(N).astype(np.float64),
        "val_b": np.random.randn(N).astype(np.float64),
    })
    df.loc[10, "val_b"] = np.nan
    cleaner = Clean(df)
    opt_df = cleaner.optimize_memory(verbose=False)

    assert opt_df["val_a"].dtype == np.float32
    assert opt_df["val_b"].dtype == np.float32
    # Preserves NaN
    assert np.isnan(opt_df["val_b"].iloc[10])
    assert cleaner.memory_summary["reduction_pct"] >= 45.0


def test_clean_total_clean_with_optimize_memory():
    df = pd.DataFrame({
        "age": np.array([25, 30, 45, 60, 22], dtype=np.int64),
        "salary": np.array([50000.0, 75000.0, 120000.0, 90000.0, 45000.0], dtype=np.float64),
    })
    cleaner = Clean(df)
    cleaned = cleaner.total_clean(optimize_memory=True)

    assert cleaned["age"].dtype in [np.int8, np.uint8]
    assert cleaned["salary"].dtype == np.float32


def test_mlpipeline_optimize_memory():
    df = pd.DataFrame({
        "num1": np.random.randn(100),
        "num2": np.random.randn(100),
        "cat1": ["a", "b", "a", "b"] * 25,
        "target": np.random.randint(0, 2, 100)
    })
    pipe = MLPipeline(optimize_memory=True)
    X, y = pipe.fit_transform(df, target="target", scale="standard")

    # Output features should be float32 instead of float64
    for col in X.select_dtypes(include='number').columns:
        assert X[col].dtype == np.float32 or X[col].dtype in [np.int8, np.int16, np.int32]


def test_clean_load_chunked(tmp_path):
    csv_file = tmp_path / "test_chunked.csv"
    N = 5000
    df = pd.DataFrame({
        "id": np.arange(N, dtype=np.int64),
        "score": np.random.randn(N).astype(np.float64),
        "category": ["A", "B", "C", "D", "E"] * (N // 5)
    })
    df.to_csv(csv_file, index=False)

    cleaner = Clean(str(csv_file))
    loaded_df = cleaner.load_chunked(chunksize=1000, downcast_numeric=True, auto_categorical=True)

    assert len(loaded_df) == N
    assert loaded_df["score"].dtype == np.float32
    assert loaded_df["id"].dtype in [np.int16, np.uint16, np.int32, np.uint32]
    assert isinstance(loaded_df["category"].dtype, pd.CategoricalDtype)


def test_clean_load_chunked_quantize_8bit(tmp_path):
    csv_file = tmp_path / "test_quant.csv"
    N = 2000
    df = pd.DataFrame({
        "continuous_val": np.linspace(0.0, 100.0, N, dtype=np.float64)
    })
    df.to_csv(csv_file, index=False)

    cleaner = Clean(str(csv_file))
    loaded_df = cleaner.load_chunked(chunksize=500, quantize_8bit=True)

    assert len(loaded_df) == N
    assert loaded_df["continuous_val"].dtype == np.uint8
    assert loaded_df["continuous_val"].min() == 0
    assert loaded_df["continuous_val"].max() == 255
