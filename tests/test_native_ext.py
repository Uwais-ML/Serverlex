import pytest
import numpy as np
import pandas as pd
from serverlex._native import (
    NATIVE_AVAILABLE,
    psi_single,
    psi_batch,
    psi_from_dataframes,
    corr_filter_pairs,
    corr_compute_filter,
    corr_pairs_from_df,
    _psi_fallback,
    _corr_filter_fallback,
    _corr_compute_fallback,
)
from serverlex.diagnostics.drift import DriftDetector
from serverlex.diagnostics.dataguard import DataGuard
from serverlex.modeling.selector import FeatureSelector


def test_native_available():
    assert NATIVE_AVAILABLE is True, "C++ native extension should be compiled and loaded"


def test_psi_single_parity():
    rng = np.random.default_rng(101)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(0.4, 1.2, 2000)

    py_val = _psi_fallback(ref, cur)
    cpp_val = psi_single(ref, cur)
    drift_val = DriftDetector.calculate_psi(ref, cur)

    assert abs(py_val - cpp_val) < 1e-4
    assert abs(drift_val - cpp_val) < 1e-4


def test_psi_batch_parity():
    rng = np.random.default_rng(102)
    M = 15
    ref = rng.normal(0, 1, (1000, M))
    cur = rng.normal(0.2, 1.1, (1000, M))

    py_scores = np.array([_psi_fallback(ref[:, j], cur[:, j]) for j in range(M)])
    cpp_scores = psi_batch(ref, cur)

    np.testing.assert_allclose(py_scores, cpp_scores, atol=1e-3)


def test_psi_from_dataframes():
    rng = np.random.default_rng(103)
    cols = ["a", "b", "c"]
    ref_df = pd.DataFrame(rng.normal(0, 1, (500, 3)), columns=cols)
    cur_df = pd.DataFrame(rng.normal(0.5, 1.5, (500, 3)), columns=cols)

    scores = psi_from_dataframes(ref_df, cur_df)
    assert set(scores.keys()) == set(cols)
    for c in cols:
        assert scores[c] >= 0.0


def test_corr_filter_pairs():
    corr_mat = np.array([
        [1.0, 0.92, 0.10],
        [0.92, 1.0, 0.88],
        [0.10, 0.88, 1.0],
    ], dtype=np.float64)

    py_pairs = _corr_filter_fallback(corr_mat, threshold=0.85)
    cpp_pairs = corr_filter_pairs(corr_mat, threshold=0.85)

    assert len(py_pairs) == 2
    assert len(cpp_pairs) == 2
    assert cpp_pairs[0] == (0, 1, 0.92)
    assert cpp_pairs[1] == (1, 2, 0.88)


def test_corr_compute_filter_subsample():
    rng = np.random.default_rng(104)
    N, M = 10_000, 5
    data = rng.normal(0, 1, (N, M))
    # Collinear pair: col 0 and col 4
    data[:, 4] = data[:, 0] * 0.99 + rng.normal(0, 0.01, N)

    pairs = corr_compute_filter(data, threshold=0.90, max_rows=2000)
    assert any(p[0] == 0 and p[1] == 4 for p in pairs)
    for p in pairs:
        assert p[2] >= 0.90


def test_dataguard_integration():
    rng = np.random.default_rng(105)
    N = 2000
    x1 = rng.normal(0, 1, N)
    x2 = x1 * 0.98 + rng.normal(0, 0.05, N)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": rng.normal(0, 1, N), "target": rng.integers(0, 2, N)})

    guard = DataGuard(df, target="target")
    collinear = guard.check_collinearity(threshold=0.85)
    assert len(collinear) >= 1
    pair_cols = [(p[0], p[1]) for p in collinear]
    assert ("x1", "x2") in pair_cols or ("x2", "x1") in pair_cols
