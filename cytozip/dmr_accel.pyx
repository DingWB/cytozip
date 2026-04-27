# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
dmr_accel.pyx — Cython kernels for the permutation root-mean-square (RMS)
DMR test (Methylpy / ALLCools style).

This module replaces the previous numba implementation in dmr.py.  All
hot loops are pure C with the GIL released; the per-site loop is
parallelized with OpenMP ``prange``.

Public entry point
------------------
``rms_run_sites(mc, cov, idx, group_a_n, n_permute, min_pvalue,
                max_row_count, max_total_count, n_threads=0)``

* ``mc``, ``cov`` : 2-D ``int64`` arrays of shape ``(n_cells, n_sites)``.
  The first ``group_a_n`` rows are group A, the rest group B.
* ``idx`` : 1-D ``int64`` array of column indices (sites) to test.
* Returns ``(p_values, frac_delta)`` as 1-D ``float64`` arrays of length
  ``len(idx)``.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, rand, srand, RAND_MAX
from libc.math cimport sqrt
from libc.stdint cimport int64_t
from cython.parallel cimport prange
cimport openmp

cnp.import_array()


# --------------------------------------------------------------------------
# Per-thread RNG: xorshift64*  (avoids contention on libc rand state and
# gives roughly U(0,1) values without locking).
# --------------------------------------------------------------------------
cdef inline unsigned long long _xorshift64(unsigned long long *state) nogil:
    cdef unsigned long long x = state[0]
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    state[0] = x
    return x * <unsigned long long> 2685821657736338717ULL


cdef inline double _u01(unsigned long long *state) nogil:
    # Top 53 bits -> [0, 1).
    return ((<double> (_xorshift64(state) >> 11)) / 9007199254740992.0)


# --------------------------------------------------------------------------
# RMS statistic on an n x 2 contingency table stored as a flat int64 buffer.
# Layout: table[i*2 + 0] = mc count, table[i*2 + 1] = unmeth count.
# --------------------------------------------------------------------------
cdef inline double _rms_stat_flat(int64_t *table, int n, int64_t m) nogil:
    cdef double row0 = 0.0, row1 = 0.0
    cdef int i
    cdef double col_i, e0, e1, d, s
    for i in range(n):
        row0 += <double> table[i * 2]
        row1 += <double> table[i * 2 + 1]
    s = 0.0
    for i in range(n):
        col_i = <double> (table[i * 2] + table[i * 2 + 1])
        e0 = col_i * row0 / <double> m
        e1 = col_i * row1 / <double> m
        d = <double> table[i * 2] - e0
        s += d * d
        d = <double> table[i * 2 + 1] - e1
        s += d * d
    return sqrt(s / (2.0 * <double> n))


cdef inline void _rms_expected(int64_t *table, int n, int64_t m,
                               double *p_out) nogil:
    """Fill ``p_out[2*n]`` with the per-cell expected probability under
    the row/col-sum independence null."""
    cdef double row0 = 0.0, row1 = 0.0
    cdef int i
    cdef double col_i
    for i in range(n):
        row0 += <double> table[i * 2]
        row1 += <double> table[i * 2 + 1]
    for i in range(n):
        col_i = <double> (table[i * 2] + table[i * 2 + 1])
        p_out[i * 2] = col_i * row0 / (<double> m * <double> m)
        p_out[i * 2 + 1] = col_i * row1 / (<double> m * <double> m)


cdef inline void _rms_permute_table(double *cum, int n, int64_t m,
                                     int64_t *out, unsigned long long *state) nogil:
    """Sample n x 2 multinomial table with cumulative-prob ``cum[2*n]``
    and total ``m`` into ``out[2*n]`` (zero-initialized inside)."""
    cdef int twoN = 2 * n
    cdef int k, lo, hi, mid
    cdef int64_t t
    cdef double u
    for k in range(twoN):
        out[k] = 0
    for t in range(m):
        u = _u01(state)
        lo = 0
        hi = twoN - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        out[lo] += 1


cdef inline double _rms_pvalue_flat(int64_t *table, int n, int64_t m,
                                     int n_permute, double min_pvalue,
                                     double *p_buf, double *cum_buf,
                                     int64_t *perm_buf,
                                     unsigned long long *state) nogil:
    """Permutation p-value with early stopping."""
    cdef double real_s, s, cs
    cdef int it, k, twoN = 2 * n
    cdef int greater = 1
    cdef double max_greater
    if m <= 0 or n < 2:
        return 1.0
    real_s = _rms_stat_flat(table, n, m)
    _rms_expected(table, n, m, p_buf)
    cs = 0.0
    for k in range(twoN):
        cs += p_buf[k]
        cum_buf[k] = cs
    max_greater = <double> n_permute * min_pvalue
    for it in range(n_permute):
        _rms_permute_table(cum_buf, n, m, perm_buf, state)
        s = _rms_stat_flat(perm_buf, n, m)
        if s >= real_s:
            greater += 1
            if <double> greater > max_greater:
                return <double> greater / <double> (it + 2)
    return <double> greater / <double> n_permute


cdef inline void _downsample_row(int64_t *row, int64_t max_count) nogil:
    cdef int64_t s = row[0] + row[1]
    cdef int64_t a, b
    if s <= max_count:
        return
    # round-half-away-from-zero: int(x + 0.5) for non-negative x
    a = <int64_t> ((<double> row[0] * <double> max_count) / <double> s + 0.5)
    b = max_count - a
    if a < 0:
        a = 0
    if b < 0:
        b = 0
    row[0] = a
    row[1] = b


cdef inline void _process_one_site(
        int64_t *mc_ptr, int64_t *cov_ptr,
        Py_ssize_t mc_stride1, Py_ssize_t cov_stride1,
        Py_ssize_t s, Py_ssize_t n_cells, int group_a_n,
        int n_permute, double min_pvalue,
        int64_t max_row_count, int64_t max_total_count,
        int64_t *t_tab, double *t_pbuf, double *t_cum, int64_t *t_perm,
        unsigned long long *t_rng,
        double *out_p, double *out_da) nogil:
    cdef int rows = 0
    cdef int n_fa = 0, n_fb = 0
    cdef int64_t c, mm, uu, total = 0
    cdef double sum_fa = 0.0, sum_fb = 0.0, scale, f
    cdef Py_ssize_t i
    for i in range(n_cells):
        c = cov_ptr[i * cov_stride1 + s]
        if c <= 0:
            continue
        mm = mc_ptr[i * mc_stride1 + s]
        uu = c - mm
        if mm < 0:
            mm = 0
        if uu < 0:
            uu = 0
        t_tab[rows * 2] = mm
        t_tab[rows * 2 + 1] = uu
        _downsample_row(t_tab + rows * 2, max_row_count)
        total += t_tab[rows * 2] + t_tab[rows * 2 + 1]
        f = <double> mm / <double> c
        if i < group_a_n:
            sum_fa = sum_fa + f
            n_fa = n_fa + 1
        else:
            sum_fb = sum_fb + f
            n_fb = n_fb + 1
        rows = rows + 1
    if rows < 2 or n_fa < 1 or n_fb < 1:
        out_p[0] = 1.0
        out_da[0] = 0.0
        return
    if total > max_total_count and total > 0:
        scale = <double> max_total_count / <double> total
        total = 0
        for i in range(rows):
            t_tab[i * 2] = <int64_t> (<double> t_tab[i * 2] * scale + 0.5)
            t_tab[i * 2 + 1] = <int64_t> (<double> t_tab[i * 2 + 1] * scale + 0.5)
            total += t_tab[i * 2] + t_tab[i * 2 + 1]
    out_p[0] = _rms_pvalue_flat(t_tab, rows, total,
                                 n_permute, min_pvalue,
                                 t_pbuf, t_cum, t_perm, t_rng)
    out_da[0] = (sum_fa / <double> n_fa) - (sum_fb / <double> n_fb)


# --------------------------------------------------------------------------
# Public entry point.
# --------------------------------------------------------------------------
def rms_run_sites(cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] mc not None,
                  cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] cov not None,
                  cnp.ndarray[cnp.int64_t, ndim=1, mode='c'] idx not None,
                  int group_a_n,
                  int n_permute=3000,
                  double min_pvalue=0.01,
                  int64_t max_row_count=50,
                  int64_t max_total_count=3000,
                  int n_threads=0,
                  unsigned long long seed=0):
    """Run permutation RMS test at every site listed in ``idx``.

    Parameters mirror the previous numba implementation.

    Notes
    -----
    * Per-thread scratch buffers (``table``, expected probabilities,
      cumulative probabilities, permutation buffer) are allocated once
      with size ``2 * n_cells``; they're indexed by ``omp_get_thread_num``.
    * Each thread also has its own xorshift64* RNG seeded from ``seed``
      (or from a fixed constant if ``seed == 0``) plus the thread id, so
      runs are reproducible.
    """
    cdef Py_ssize_t n_cells = mc.shape[0]
    cdef Py_ssize_t n_test = idx.shape[0]
    cdef Py_ssize_t k, i
    cdef int tid
    cdef int max_threads

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] out_p = \
        np.empty(n_test, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] out_da = \
        np.empty(n_test, dtype=np.float64)
    cdef double *out_p_ptr = <double *> out_p.data
    cdef double *out_da_ptr = <double *> out_da.data

    cdef int64_t *mc_ptr = <int64_t *> mc.data
    cdef int64_t *cov_ptr = <int64_t *> cov.data
    cdef int64_t *idx_ptr = <int64_t *> idx.data
    cdef Py_ssize_t mc_stride1 = mc.strides[0] // sizeof(int64_t)
    cdef Py_ssize_t cov_stride1 = cov.strides[0] // sizeof(int64_t)
    # mc / cov are C-contiguous (mode='c'), so strides[0] == n_sites and
    # strides[1] == 1.  We index as [row * stride0 + col].

    if n_threads <= 0:
        max_threads = openmp.omp_get_max_threads()
    else:
        max_threads = n_threads
    if max_threads < 1:
        max_threads = 1

    # Per-thread scratch buffers, sized for the worst case (all cells covered).
    cdef Py_ssize_t buf_n = 2 * n_cells
    cdef int64_t *table_buf = <int64_t *> malloc(
        max_threads * buf_n * sizeof(int64_t))
    cdef double *p_buf = <double *> malloc(
        max_threads * buf_n * sizeof(double))
    cdef double *cum_buf = <double *> malloc(
        max_threads * buf_n * sizeof(double))
    cdef int64_t *perm_buf = <int64_t *> malloc(
        max_threads * buf_n * sizeof(int64_t))
    cdef unsigned long long *rng_state = <unsigned long long *> malloc(
        max_threads * sizeof(unsigned long long))
    if (table_buf == NULL or p_buf == NULL or cum_buf == NULL
            or perm_buf == NULL or rng_state == NULL):
        if table_buf != NULL: free(table_buf)
        if p_buf != NULL: free(p_buf)
        if cum_buf != NULL: free(cum_buf)
        if perm_buf != NULL: free(perm_buf)
        if rng_state != NULL: free(rng_state)
        raise MemoryError("dmr_accel: scratch allocation failed")

    cdef unsigned long long base_seed = seed if seed != 0 else \
        <unsigned long long> 0x9E3779B97F4A7C15ULL
    for i in range(max_threads):
        rng_state[i] = base_seed ^ (<unsigned long long> i + 1) * \
            <unsigned long long> 0xBF58476D1CE4E5B9ULL

    cdef int64_t *t_tab
    cdef double *t_pbuf
    cdef double *t_cum
    cdef int64_t *t_perm
    cdef unsigned long long *t_rng
    cdef Py_ssize_t s

    with nogil:
        # Dynamic schedule: per-site work is highly uneven because the
        # permutation test early-stops on non-significant sites
        # (a few iters) but runs full ``n_permute`` on significant ones.
        # Static would leave threads idle near the end; dynamic keeps
        # them all busy with chunks of 16 sites.
        for k in prange(n_test, num_threads=max_threads,
                        schedule='dynamic', chunksize=16):
            tid = openmp.omp_get_thread_num()
            t_tab = table_buf + tid * buf_n
            t_pbuf = p_buf + tid * buf_n
            t_cum = cum_buf + tid * buf_n
            t_perm = perm_buf + tid * buf_n
            t_rng = rng_state + tid

            s = idx_ptr[k]
            _process_one_site(
                mc_ptr, cov_ptr, mc_stride1, cov_stride1,
                s, n_cells, group_a_n,
                n_permute, min_pvalue,
                max_row_count, max_total_count,
                t_tab, t_pbuf, t_cum, t_perm, t_rng,
                out_p_ptr + k, out_da_ptr + k,
            )

    free(table_buf)
    free(p_buf)
    free(cum_buf)
    free(perm_buf)
    free(rng_state)
    return out_p, out_da
