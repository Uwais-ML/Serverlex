"""
serverlex._native
─────────────────────────────────────────────────────────────────────────────
Thin Python wrapper around the optional C++ extension (_serverlex_ext).

Kernel 1 — PSI batch:
  psi_single(ref, cur)                     → float
  psi_batch(ref_arr, cur_arr)              → np.ndarray shape (M,)
  psi_from_dataframes(ref_df, cur_df, ...) → dict[str, float]

Kernel 2 — Pairwise Pearson correlation:
  corr_filter_pairs(corr_mat, threshold)              → list[(i,j,r)]
  corr_compute_filter(data, threshold, max_rows=50k)  → list[(i,j,r)]
  corr_pairs_from_df(df, cols, threshold, max_rows)   → list[(col_a,col_b,r)]

All functions fall back transparently to pure-Python NumPy when the extension
has not been compiled.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("serverlex.native")

# ─────────────────────────────────────────────────────────────────────────────
# Try loading the compiled C++ extension
# ─────────────────────────────────────────────────────────────────────────────
try:
    from serverlex import _serverlex_ext as _ext          # type: ignore[attr-defined]
    NATIVE_AVAILABLE: bool = True
    HAS_OPENMP: bool       = bool(getattr(_ext, "has_openmp", False))
    _EXT_VERSION: str      = str(getattr(_ext, "__version__", "unknown"))
    logger.info(
        "Serverlex native C++ extension loaded (v%s, OpenMP=%s)",
        _EXT_VERSION, HAS_OPENMP,
    )
except ImportError:
    _ext             = None
    NATIVE_AVAILABLE = False
    HAS_OPENMP       = False
    _EXT_VERSION     = "n/a"
    logger.debug(
        "Serverlex native extension not found — falling back to pure Python. "
        "Run `python setup_ext.py build_ext --inplace` to build it."
    )


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 1 — PSI
# ═════════════════════════════════════════════════════════════════════════════

def _psi_fallback(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Pure-Python fallback — numerically identical to DriftDetector.calculate_psi."""
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins      = np.percentile(expected, quantiles)
    bins[0]   = -np.inf
    bins[-1]  =  np.inf
    bins      = np.unique(bins)
    if len(bins) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts,   _ = np.histogram(actual,   bins=bins)
    ep = np.where(expected_counts == 0, 0.0001, expected_counts) / len(expected)
    ap = np.where(actual_counts   == 0, 0.0001, actual_counts)   / len(actual)
    return float(np.round(np.sum((ap - ep) * np.log(ap / ep)), 4))


def psi_single(ref: np.ndarray, cur: np.ndarray, num_bins: int = 10) -> float:
    """Single-column PSI — drop-in for DriftDetector.calculate_psi()."""
    if NATIVE_AVAILABLE:
        return float(_ext.psi_single(
            np.asarray(ref, dtype=np.float64),
            np.asarray(cur, dtype=np.float64),
            num_bins,
        ))
    return _psi_fallback(ref, cur, num_bins)


def psi_batch(ref_arr: np.ndarray, cur_arr: np.ndarray, num_bins: int = 10) -> np.ndarray:
    """
    Compute PSI for every column of two 2-D float64 arrays.

    Parameters
    ----------
    ref_arr : ndarray shape (N_ref, M)
    cur_arr : ndarray shape (N_cur, M)

    Returns
    -------
    ndarray shape (M,) — PSI per column.
    """
    ref_arr = np.asarray(ref_arr, dtype=np.float64)
    cur_arr = np.asarray(cur_arr, dtype=np.float64)
    if ref_arr.ndim == 1: ref_arr = ref_arr.reshape(-1, 1)
    if cur_arr.ndim == 1: cur_arr = cur_arr.reshape(-1, 1)
    if NATIVE_AVAILABLE:
        return _ext.psi_batch(ref_arr, cur_arr, num_bins)
    M = ref_arr.shape[1]
    result = np.zeros(M, dtype=np.float64)
    for j in range(M):
        result[j] = _psi_fallback(ref_arr[:, j], cur_arr[:, j], num_bins)
    return result


def psi_from_dataframes(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    num_bins: int = 10,
) -> Dict[str, float]:
    """Compute PSI for numeric columns shared by two DataFrames."""
    if columns is None:
        columns = [
            c for c in ref_df.columns
            if c in cur_df.columns
            and pd.api.types.is_numeric_dtype(ref_df[c])
            and pd.api.types.is_numeric_dtype(cur_df[c])
        ]
    if not columns:
        return {}
    scores = psi_batch(
        ref_df[columns].to_numpy(dtype=np.float64),
        cur_df[columns].to_numpy(dtype=np.float64),
        num_bins,
    )
    return dict(zip(columns, scores.tolist()))


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 2 — Pairwise Pearson Correlation
# ═════════════════════════════════════════════════════════════════════════════

def _corr_filter_fallback(
    corr_mat: np.ndarray, threshold: float
) -> List[Tuple[int, int, float]]:
    """Pure-Python fallback for corr_filter_pairs."""
    M = corr_mat.shape[0]
    pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            r = corr_mat[i, j]
            if not np.isnan(r) and r >= threshold:
                pairs.append((i, j, round(float(r), 4)))
    return pairs


def _corr_compute_fallback(
    data: np.ndarray, threshold: float, max_rows: int = 50_000
) -> List[Tuple[int, int, float]]:
    """Pure-Python fallback for corr_compute_filter (with subsampling)."""
    N, M = data.shape
    if N > max_rows:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(N, size=max_rows, replace=False)
        data = data[idx]
    corr = np.corrcoef(data.T)          # (M, M)
    corr = np.abs(corr)
    return _corr_filter_fallback(corr, threshold)


def corr_filter_pairs(
    corr_mat: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """
    Filter a pre-computed M×M correlation matrix for pairs above *threshold*.

    Replaces the Python nested loop + ``.loc[]`` in DataGuard / FeatureSelector.

    Parameters
    ----------
    corr_mat  : ndarray shape (M, M), float64  — e.g. df.corr().abs().to_numpy()
    threshold : float

    Returns
    -------
    list of (i, j, r) tuples, upper triangle only.
    """
    corr_mat = np.asarray(corr_mat, dtype=np.float64)
    if NATIVE_AVAILABLE:
        return list(_ext.corr_filter_pairs(corr_mat, threshold))
    return _corr_filter_fallback(corr_mat, threshold)


def corr_compute_filter(
    data: np.ndarray,
    threshold: float,
    max_rows: int = 50_000,
    seed: int = 42,
) -> List[Tuple[int, int, float]]:
    """
    Compute Pearson correlation from raw data with subsampling, then filter.

    This is the key capability pandas ``.corr()`` cannot replicate natively:
    subsample rows *before* computing, so 1 M-row datasets run 20× faster
    with negligible statistical loss.

    Parameters
    ----------
    data      : ndarray shape (N, M), float64
    threshold : float
    max_rows  : int, default 50_000
    seed      : int, default 42

    Returns
    -------
    list of (i, j, r) tuples, upper triangle only.
    """
    data = np.asarray(data, dtype=np.float64)
    if NATIVE_AVAILABLE:
        return list(_ext.corr_compute_filter(data, threshold, max_rows, seed))
    return _corr_compute_fallback(data, threshold, max_rows)


def corr_pairs_from_df(
    df: pd.DataFrame,
    cols: List[str],
    threshold: float,
    max_rows: int = 50_000,
    use_subsample: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    Convenience wrapper: return collinear column-name pairs from a DataFrame.

    Parameters
    ----------
    df           : DataFrame
    cols         : list of numeric column names to check
    threshold    : float — minimum |r|
    max_rows     : int — subsample budget (only used when use_subsample=True)
    use_subsample: bool — use the subsampled C++ compute path (default True).
                   Set False to pass a pre-computed corr matrix instead
                   (useful when .corr() was already called).

    Returns
    -------
    list of (col_a: str, col_b: str, r: float) tuples.
    """
    if len(cols) < 2:
        return []

    data = df[cols].to_numpy(dtype=np.float64)

    if use_subsample:
        idx_pairs = corr_compute_filter(data, threshold, max_rows)
    else:
        corr_mat = np.abs(np.corrcoef(data.T))
        idx_pairs = corr_filter_pairs(corr_mat, threshold)

    return [(cols[i], cols[j], r) for i, j, r in idx_pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Introspection
# ─────────────────────────────────────────────────────────────────────────────
def native_info() -> dict:
    """Return a summary dict of the native-extension status."""
    return {
        "native_available": NATIVE_AVAILABLE,
        "has_openmp":       HAS_OPENMP,
        "ext_version":      _EXT_VERSION,
        "backend": "C++ (_serverlex_ext)" if NATIVE_AVAILABLE else "Python (NumPy fallback)",
    }
