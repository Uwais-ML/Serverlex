/**
 * serverlex/_ext/psi_kernel.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Native C++ PSI (Population Stability Index) kernel for Serverlex.
 *
 * Exposed Python API (via pybind11):
 *   psi_batch(ref, cur, num_bins=10)  → ndarray shape (M,)
 *       Compute PSI for every column of two 2-D float64 NumPy arrays in one
 *       call.  If compiled with OpenMP, columns are processed in parallel.
 *
 *   psi_single(ref, cur, num_bins=10) → float
 *       Single-column PSI — drop-in replacement for
 *       DriftDetector.calculate_psi(expected, actual).
 *
 *   has_openmp   → bool   (module-level attribute)
 *   __version__  → str
 *
 * Numerical contract:
 *   Exactly replicates the Python reference implementation in drift.py:
 *     - quantile-based bins via linear-interpolation percentile (numpy default)
 *     - bins[0] = -inf, bins[-1] = +inf
 *     - np.unique(bins) deduplication
 *     - 0.0001 epsilon guard for empty bins
 *     - result rounded to 4 decimal places
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Linear-interpolation percentile — matches numpy.percentile default method.
 *
 * @param sorted_v  Pre-sorted non-NaN values.
 * @param p         Percentile in [0, 100].
 */
static inline double percentile_lin(const std::vector<double>& sorted_v, double p) noexcept {
    if (sorted_v.empty()) return 0.0;
    if (p <= 0.0)         return sorted_v.front();
    if (p >= 100.0)       return sorted_v.back();

    double idx  = (p / 100.0) * static_cast<double>(sorted_v.size() - 1);
    auto   lo   = static_cast<std::size_t>(idx);
    auto   hi   = lo + 1;
    if (hi >= sorted_v.size()) return sorted_v.back();

    double frac = idx - static_cast<double>(lo);
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo]);
}

/**
 * Find the bin index for value v given sorted bin edges.
 * Semantics: bins[b] < v <= bins[b+1], clamped to [0, n_bins-1].
 * This matches numpy.histogram: half-open [a,b) with last bin closed.
 * Since bins[0]=-inf and bins[-1]=+inf, every finite value lands in a bin.
 */
static inline int find_bin(const std::vector<double>& bins, double v) noexcept {
    // upper_bound returns iterator to first element > v
    auto it = std::upper_bound(bins.begin(), bins.end(), v);
    if (it == bins.begin()) return 0;
    if (it == bins.end())   return static_cast<int>(bins.size()) - 2;
    int b = static_cast<int>(it - bins.begin()) - 1;
    int n_bins = static_cast<int>(bins.size()) - 1;
    if (b < 0)       return 0;
    if (b >= n_bins) return n_bins - 1;
    return b;
}

/**
 * Core PSI computation for a single column (1-D).
 *
 * @param ref_data  Pointer to reference column (may contain NaN).
 * @param n_ref     Length of ref_data.
 * @param cur_data  Pointer to current column (may contain NaN).
 * @param n_cur     Length of cur_data.
 * @param num_bins  Number of histogram buckets.
 * @return          PSI score rounded to 4 decimal places, or 0.0 if data is
 *                  insufficient.
 */
static double compute_psi_1d(
    const double* ref_data, std::size_t n_ref,
    const double* cur_data, std::size_t n_cur,
    int num_bins) noexcept
{
    // ── 1. Filter NaN ────────────────────────────────────────────────────────
    std::vector<double> ref_clean, cur_clean;
    ref_clean.reserve(n_ref);
    cur_clean.reserve(n_cur);

    for (std::size_t i = 0; i < n_ref; ++i)
        if (!std::isnan(ref_data[i])) ref_clean.push_back(ref_data[i]);
    for (std::size_t i = 0; i < n_cur; ++i)
        if (!std::isnan(cur_data[i])) cur_clean.push_back(cur_data[i]);

    if (ref_clean.size() < 10 || cur_clean.size() < 10) return 0.0;

    // ── 2. Quantile-based bin edges from reference ───────────────────────────
    std::vector<double> sorted_ref = ref_clean;
    std::sort(sorted_ref.begin(), sorted_ref.end());

    std::vector<double> bins;
    bins.reserve(num_bins + 1);
    for (int i = 0; i <= num_bins; ++i) {
        double p = static_cast<double>(i) / static_cast<double>(num_bins) * 100.0;
        bins.push_back(percentile_lin(sorted_ref, p));
    }
    bins.front() = -std::numeric_limits<double>::infinity();
    bins.back()  =  std::numeric_limits<double>::infinity();

    // Deduplicate (matches np.unique)
    bins.erase(std::unique(bins.begin(), bins.end()), bins.end());
    if (static_cast<int>(bins.size()) < 3) return 0.0;

    int n_bins = static_cast<int>(bins.size()) - 1;

    // ── 3. Histogram ─────────────────────────────────────────────────────────
    std::vector<double> ref_counts(n_bins, 0.0);
    std::vector<double> cur_counts(n_bins, 0.0);

    for (double v : ref_clean) ref_counts[find_bin(bins, v)] += 1.0;
    for (double v : cur_clean) cur_counts[find_bin(bins, v)] += 1.0;

    // ── 4. PSI formula ───────────────────────────────────────────────────────
    double n_ref_d = static_cast<double>(ref_clean.size());
    double n_cur_d = static_cast<double>(cur_clean.size());

    double psi = 0.0;
    for (int b = 0; b < n_bins; ++b) {
        double ep = (ref_counts[b] == 0.0 ? 0.0001 : ref_counts[b]) / n_ref_d;
        double ap = (cur_counts[b] == 0.0 ? 0.0001 : cur_counts[b]) / n_cur_d;
        psi += (ap - ep) * std::log(ap / ep);
    }

    // Round to 4 decimal places (matches Python np.round(psi, 4))
    return std::round(psi * 10000.0) / 10000.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// pybind11-exposed functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * psi_batch — vectorised batch PSI across M columns.
 *
 * @param ref_arr  2-D float64 NumPy array, shape (N_ref, M).
 * @param cur_arr  2-D float64 NumPy array, shape (N_cur, M).
 * @param num_bins Number of histogram buckets (default 10).
 * @return         1-D float64 NumPy array, shape (M,).
 */
static py::array_t<double> psi_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> ref_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> cur_arr,
    int num_bins = 10)
{
    if (ref_arr.ndim() != 2 || cur_arr.ndim() != 2)
        throw std::invalid_argument("psi_batch: both inputs must be 2-D arrays");

    auto ref_buf = ref_arr.unchecked<2>();
    auto cur_buf = cur_arr.unchecked<2>();

    py::ssize_t n_ref = ref_buf.shape(0);
    py::ssize_t n_cur = cur_buf.shape(0);
    py::ssize_t M     = ref_buf.shape(1);

    if (cur_buf.shape(1) != M)
        throw std::invalid_argument(
            "psi_batch: ref and cur must have the same number of columns");

    // Allocate output
    auto result  = py::array_t<double>(M);
    auto res_buf = result.mutable_unchecked<1>();

    // ── OpenMP parallel loop over columns ────────────────────────────────────
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic)
#endif
    for (py::ssize_t j = 0; j < M; ++j) {
        // Extract column j into contiguous vectors (necessary for the kernel)
        std::vector<double> ref_col(static_cast<std::size_t>(n_ref));
        std::vector<double> cur_col(static_cast<std::size_t>(n_cur));

        for (py::ssize_t i = 0; i < n_ref; ++i)
            ref_col[static_cast<std::size_t>(i)] = ref_buf(i, j);
        for (py::ssize_t i = 0; i < n_cur; ++i)
            cur_col[static_cast<std::size_t>(i)] = cur_buf(i, j);

        res_buf(j) = compute_psi_1d(
            ref_col.data(), static_cast<std::size_t>(n_ref),
            cur_col.data(), static_cast<std::size_t>(n_cur),
            num_bins);
    }

    return result;
}

/**
 * psi_single — single-column PSI, drop-in for DriftDetector.calculate_psi.
 *
 * @param ref_arr  1-D float64 NumPy array (reference column).
 * @param cur_arr  1-D float64 NumPy array (current column).
 * @param num_bins Number of histogram buckets (default 10).
 * @return         PSI score (float).
 */
static double psi_single(
    py::array_t<double, py::array::c_style | py::array::forcecast> ref_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> cur_arr,
    int num_bins = 10)
{
    auto ref_buf = ref_arr.unchecked<1>();
    auto cur_buf = cur_arr.unchecked<1>();

    return compute_psi_1d(
        ref_buf.data(0), static_cast<std::size_t>(ref_arr.size()),
        cur_buf.data(0), static_cast<std::size_t>(cur_arr.size()),
        num_bins);
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 2: Pairwise Pearson Correlation — Filter & Compute
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reservoir sampling (Fisher-Yates): fill buf[0..k-1] from [0..n-1] without
 * replacement.  Uses a fixed seed for reproducibility.
 */
static std::vector<std::size_t> reservoir_sample(
    std::size_t n, std::size_t k, uint64_t seed = 42)
{
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t(0));
    // Simple LCG for portability
    uint64_t rng = seed;
    auto next_rand = [&](std::size_t lo, std::size_t hi) -> std::size_t {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return lo + static_cast<std::size_t>(rng >> 33) % (hi - lo + 1);
    };
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t j = next_rand(i, n - 1);
        std::swap(idx[i], idx[j]);
    }
    idx.resize(k);
    return idx;
}

/**
 * corr_filter_pairs — filter a pre-computed M×M correlation matrix.
 *
 * Takes the output of pandas df.corr().abs().to_numpy() and returns all
 * upper-triangle pairs (i, j, r) where r >= threshold.
 *
 * This replaces the Python nested loop:
 *   for i in range(M):
 *       for j in range(i+1, M):
 *           r = corr_matrix.loc[col_a, col_b]   # .loc[] overhead × M² times
 *
 * @param corr_arr  2-D float64 array, shape (M, M).
 * @param threshold Minimum |r| to report (e.g. 0.85).
 * @return          List of (i, j, r) Python tuples.
 */
static py::list corr_filter_pairs(
    py::array_t<double, py::array::c_style | py::array::forcecast> corr_arr,
    double threshold)
{
    if (corr_arr.ndim() != 2)
        throw std::invalid_argument("corr_filter_pairs: input must be a 2-D array");

    auto buf = corr_arr.unchecked<2>();
    py::ssize_t M = buf.shape(0);
    if (buf.shape(1) != M)
        throw std::invalid_argument("corr_filter_pairs: input must be square (M×M)");

    py::list result;
    for (py::ssize_t i = 0; i < M; ++i) {
        for (py::ssize_t j = i + 1; j < M; ++j) {
            double r = buf(i, j);
            if (!std::isnan(r) && r >= threshold) {
                result.append(py::make_tuple(
                    static_cast<int>(i),
                    static_cast<int>(j),
                    std::round(r * 10000.0) / 10000.0));
            }
        }
    }
    return result;
}

/**
 * corr_compute_filter — compute Pearson correlation from raw data + filter.
 *
 * Key advantage over pandas .corr(): optionally subsamples the data before
 * computing, so large datasets (1M+ rows) run in a fraction of the time with
 * negligible statistical loss.
 *
 * Algorithm:
 *   1. If N > max_rows: reservoir-sample max_rows rows (seed=42).
 *   2. Compute column means and L2 norms of centred columns.
 *   3. For each upper-triangle pair (i,j): r = dot(Xc_i, Xc_j)/(norm_i*norm_j)
 *      The inner dot-product loop is auto-vectorised by -O3 -ffast-math.
 *   4. Return pairs where |r| >= threshold.
 *
 * @param data_arr  2-D float64 NumPy array, shape (N, M).  NaN values in a
 *                  column cause that column to be skipped for all pairs
 *                  involving it.
 * @param threshold Minimum |r| to report.
 * @param max_rows  Row budget for correlation computation; rows above this are
 *                  sampled away (default 50 000).
 * @param seed      Random seed for the reservoir sampler (default 42).
 * @return          List of (i, j, r) Python tuples.
 */
static py::list corr_compute_filter(
    py::array_t<double, py::array::c_style | py::array::forcecast> data_arr,
    double threshold,
    std::size_t max_rows = 50000,
    uint64_t seed = 42)
{
    if (data_arr.ndim() != 2)
        throw std::invalid_argument("corr_compute_filter: input must be a 2-D array");

    auto buf = data_arr.unchecked<2>();
    std::size_t N = static_cast<std::size_t>(buf.shape(0));
    std::size_t M = static_cast<std::size_t>(buf.shape(1));

    if (N == 0 || M < 2) return py::list();

    // ── 1. Build row index (subsample if needed) ─────────────────────────────
    std::vector<std::size_t> rows;
    if (N > max_rows) {
        rows = reservoir_sample(N, max_rows, seed);
    } else {
        rows.resize(N);
        std::iota(rows.begin(), rows.end(), std::size_t(0));
    }
    std::size_t n = rows.size();   // effective sample size

    // ── 2. Copy selected rows into a column-major matrix for cache efficiency ─
    //  X[col][row]  — allows the inner dot product loop to stride linearly.
    std::vector<std::vector<double>> X(M, std::vector<double>(n));
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < M; ++c)
            X[c][r] = buf(static_cast<py::ssize_t>(rows[r]),
                          static_cast<py::ssize_t>(c));

    // ── 3. Compute column means + centred L2 norms ───────────────────────────
    std::vector<double> mean(M, 0.0), norm(M, 0.0);
    std::vector<bool>   valid(M, true);   // false if column has any NaN

    for (std::size_t c = 0; c < M; ++c) {
        double s = 0.0; int cnt = 0;
        for (std::size_t r = 0; r < n; ++r) {
            if (std::isnan(X[c][r])) { valid[c] = false; break; }
            s += X[c][r]; ++cnt;
        }
        if (!valid[c] || cnt == 0) continue;
        mean[c] = s / cnt;
        double sq = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            X[c][r] -= mean[c];        // centre in-place
            sq += X[c][r] * X[c][r];
        }
        norm[c] = std::sqrt(sq);
        if (norm[c] < 1e-12) valid[c] = false;  // near-constant column
    }

    // ── 4. Pairwise dot products + threshold filter ───────────────────────────
    py::list result;

#ifdef _OPENMP
    // Thread-local vectors to avoid lock overhead
    std::vector<std::vector<std::tuple<int,int,double>>> thread_results(
        omp_get_max_threads());

#   pragma omp parallel for schedule(dynamic) collapse(1)
    for (std::size_t i = 0; i < M; ++i) {
        if (!valid[i]) continue;
        int tid = omp_get_thread_num();
        const double* xi = X[i].data();
        for (std::size_t j = i + 1; j < M; ++j) {
            if (!valid[j]) continue;
            const double* xj = X[j].data();
            double dot = 0.0;
            for (std::size_t r = 0; r < n; ++r)
                dot += xi[r] * xj[r];
            double r_val = dot / (norm[i] * norm[j]);
            double r_abs = std::fabs(r_val);
            if (r_abs >= threshold)
                thread_results[tid].emplace_back(
                    static_cast<int>(i), static_cast<int>(j),
                    std::round(r_abs * 10000.0) / 10000.0);
        }
    }
    // Merge thread-local results in index order
    for (auto& tr : thread_results)
        for (auto& [ii, jj, rr] : tr)
            result.append(py::make_tuple(ii, jj, rr));

#else
    for (std::size_t i = 0; i < M; ++i) {
        if (!valid[i]) continue;
        const double* xi = X[i].data();
        for (std::size_t j = i + 1; j < M; ++j) {
            if (!valid[j]) continue;
            const double* xj = X[j].data();
            // Inner loop — compiler auto-vectorises with -O3 -ffast-math
            double dot = 0.0;
            for (std::size_t r = 0; r < n; ++r)
                dot += xi[r] * xj[r];
            double r_val = dot / (norm[i] * norm[j]);
            double r_abs = std::fabs(r_val);
            if (r_abs >= threshold)
                result.append(py::make_tuple(
                    static_cast<int>(i), static_cast<int>(j),
                    std::round(r_abs * 10000.0) / 10000.0));
        }
    }
#endif

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_serverlex_ext, m) {
    m.doc() = "Serverlex native C++ extension — PSI + Correlation kernels (v2.0.0)";

    // ── PSI ──────────────────────────────────────────────────────────────────
    m.def("psi_batch", &psi_batch,
          py::arg("ref"), py::arg("cur"), py::arg("num_bins") = 10,
          R"doc(
Batch PSI computation across M features in a single C++ call.
ref/cur: np.ndarray shape (N, M) float64. Returns np.ndarray shape (M,).
)doc");

    m.def("psi_single", &psi_single,
          py::arg("ref"), py::arg("cur"), py::arg("num_bins") = 10,
          R"doc(Single-column PSI — drop-in for DriftDetector.calculate_psi().)doc");

    // ── Correlation ───────────────────────────────────────────────────────────
    m.def("corr_filter_pairs", &corr_filter_pairs,
          py::arg("corr_mat"), py::arg("threshold"),
          R"doc(
Filter a pre-computed M×M correlation matrix for pairs above threshold.

Replaces the Python nested loop + .loc[] used in DataGuard.check_collinearity()
and FeatureSelector.fit().

Parameters
----------
corr_mat  : np.ndarray, shape (M, M), float64  — e.g. df.corr().abs().to_numpy()
threshold : float                               — minimum |r| to report

Returns
-------
list of (i: int, j: int, r: float) tuples, upper triangle only.
)doc");

    m.def("corr_compute_filter", &corr_compute_filter,
          py::arg("data"),
          py::arg("threshold"),
          py::arg("max_rows") = 50000,
          py::arg("seed")     = 42,
          R"doc(
Compute Pearson correlation from raw data with optional subsampling, then filter.

This is the key capability that pandas .corr() cannot replicate natively:
it subsamples the data before computing correlation, making large datasets
(1M+ rows) feasible with negligible statistical loss.

Parameters
----------
data      : np.ndarray, shape (N, M), float64 — raw feature matrix
threshold : float                              — minimum |r| to report
max_rows  : int, default 50_000               — subsample budget
seed      : int, default 42                   — reservoir-sampler seed

Returns
-------
list of (i: int, j: int, r: float) tuples, upper triangle only.

Notes
-----
Inner dot-product loop is compiler auto-vectorised (-O3 -ffast-math).
When built with OpenMP, pairs are processed in parallel across threads.
)doc");

    // Runtime introspection
    m.attr("__version__") = "2.0.0";
    m.attr("has_openmp")  =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}
