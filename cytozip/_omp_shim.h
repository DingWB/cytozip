/* OpenMP shim for the cytozip Cython extensions.
 *
 * The extensions use ``prange`` + ``omp_get_max_threads`` /
 * ``omp_get_thread_num``. When the compiler supports OpenMP (gcc, conda
 * clang + llvm-openmp, Homebrew libomp) ``_OPENMP`` is defined and the real
 * <omp.h> is used. Apple's stock clang has no OpenMP: in that case Cython
 * lowers ``prange`` to a plain serial loop and this header supplies the two
 * runtime calls so the same .pyx still compiles (single-threaded).
 */
#ifndef CYTOZIP_OMP_SHIM_H
#define CYTOZIP_OMP_SHIM_H

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
#endif

#endif /* CYTOZIP_OMP_SHIM_H */
