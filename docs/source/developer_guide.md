# Developer Guide

A high-level map of the cytozip codebase for contributors. The goal of this
page is to make it possible to find your way around the package without
reading every line — for the API reference see the *modules* page.

## Architecture overview

A `.cz` file is a columnar, chunk-based binary format for tabular genomics
data (e.g. `allc` methylation tables). The on-disk layout is:

```
[Header] [Chunk 0] [Chunk 1] ... [Chunk N] [ChunkIndex] [EOF marker]
```

* **Header** — magic, version, total size, message, per-column format
  strings (`struct`-compatible), column names, chunk-key names.
* **Chunk** — one logical group of records that share the same chunk-key
  values (e.g. a single chromosome). Internally split into independently
  compressed *blocks* of at most 256 KiB raw bytes (raw DEFLATE).
* **Chunk tail** — appended after the last block of a chunk; contains
  the per-block virtual offsets and chunk-key values.
* **Chunk index** — appended at end-of-file before the EOF marker so HTTP
  / remote readers can locate any chunk in O(1) without scanning the
  whole file.
* **EOF marker** — 28-byte sentinel borrowed from BGZF.

A *virtual offset* packs `(block_start << 20) | within_block_offset` into
a single 64-bit integer (44 bits file offset, 20 bits within-block),
providing O(1) random access to any record.

Strictly-monotonic integer columns (typically genomic positions) can be
stored with **delta encoding** (`_ENC_DELTA`): within each block we store
first-order differences, so DEFLATE compresses a `pos` column to a
fraction of its raw size. Decoding is a single numpy `cumsum` per
delta column per block.

## Module map

| Module | Lines | Purpose |
| --- | --- | --- |
| `cytozip/cz.py` | ~6000 | Core `Reader` / `Writer`, the on-disk format, `catcz` / `extract` / `aggregate` / `align_cz`, remote (HTTP range) reading |
| `cytozip/cz_accel.pyx` | ~2650 | Cython accelerators (libdeflate block (de)compression with OpenMP, delta encode/decode, record packing, region queries, chunk index parser, integer text ingestion) |
| `cytozip/bam.py` | ~1370 | Pileup-based BAM → `.cz` methylation extraction (htslib Cython backend in `_bam_pileup.pyx`, `samtools mpileup` fallback) |
| `cytozip/allc.py` | ~1320 | `allc.tsv(.gz)` → `.cz`, reference building (`AllC` / `build_ref`), `extractCG`, `compare_allc` |
| `cytozip/index.py` | ~570 | Context (CGN / CHN) and region indexes over a reference; IDs are **1-based** (`Reader.get_index_rows` → 0-based) |
| `cytozip/merge.py` | ~1500 | Sum many `.cz` into pseudobulks (`merge_cz`, multi-reference variant, per-cell-type merging) |
| `cytozip/pivot.py` | ~480 | Wide per-cell fraction / one-vs-rest Fisher TSV matrices |
| `cytozip/features.py` | ~2280 | Aggregation over features → AnnData (`cz_to_anndata`, `_multiref`) and the single-cytosine sparse matrix (`cz_to_anndata_sites`); Beta-Binomial prior / posterior fraction |
| `cytozip/dmr.py` + `dmr_accel.pyx` | ~2900 | DMR calling: permutation RMS test (OpenMP), 1-vs-1 Fisher, CH binned variant, one-vs-rest, DMS→DMR merging, annotation, array DMRs |
| `cytozip/peaks.py` | ~1220 | Peak calling from unmethylated counts (MACS3 pseudo-reads route and native Poisson/bedGraph route) |
| `cytozip/dataloader.py` | ~1150 | `CzWindowLoader` / `CzWindowDataset`: streaming `(n_cells, sites)` windows for deep learning |
| `cytozip/model.py` | ~3500 | Cell-type classifier (site-likelihood naive Bayes) and reference-based bulk deconvolution |
| `cytozip/array.py` | ~360 | Methylation-array (beta) matrices → `.cz` with optional quantisation |
| `cytozip/__init__.py` | ~1720 | CLI entry point (`czip ...`) with lazy imports |

See `dev.md` (repo root) for the format specification, tutorials and
behavioural notes, and `docs/Methods.md` for the statistical methods.

### `cz.py` — core format

* **`Writer`** — open, accumulate records into the active block, flush
  full blocks (`_BLOCK_MAX_LEN = (1 << 18) - 1`, i.e. 256 KiB − 1), close out the chunk
  with its tail when the chunk-key values change, write the chunk index
  + EOF marker on `close()`.
* **`Reader`** — `mmap`s the file, parses the header + chunk index,
  exposes `fetch_chunk_bytes`, `iter_blocks`, `get_record`, region
  queries, and `to_pandas()`. Hot paths dispatch to Cython (see
  `_ensure_cz_accel`).
* **Lazy modules** — numpy / pandas are wrapped in `_LazyModule` so
  `import cytozip` and CLI subcommands like `czip header` start in
  ~60 ms instead of ~130 ms.
* **Lazy Cython accelerators** — every `_c_*` name in `cz.py` is bound
  to `None` at import time and swapped in by `_ensure_cz_accel()` on
  first use. The public `_load_bcz_block` alias is *re-bound* at that
  point so callers who captured it earlier do not get stuck on the pure-
  Python fallback.
* **mmap memory model** — `Reader.__init__` calls `mmap.mmap(fd, 0,
  PROT_READ)`. Sequential reads fault every page into RSS; for a 1.3 GB
  reference this can pin >2 GB. Two helpers are exposed:
  * `Reader.advise_sequential()` — `madvise(MADV_SEQUENTIAL)` over the
    whole map; call this once for whole-file walks.
  * `Reader.release_chunk(dim)` — finds the chunk's exact byte range
    (next chunk's offset, or EOF) and `madvise(MADV_DONTNEED)` only
    that range. Call after you finish a chunk so the kernel reclaims
    its file pages.

### `cz_accel.pyx` — Cython hot paths

When `cytozip.cz_accel` imports successfully, the following functions
move into pure C / libdeflate:

* `load_bcz_block` — read + decompress one block into a Python `bytes`.
* `compress_block` / `pack_records[_fast]` — Writer-side hot paths.
* `unpack_records` / `read_1record` / `seek_and_read_1record` —
  zero-copy record extraction.
* `query_regions[_flat]` — region-based scans.
* `parse_czix` — chunk-index parser.
* `extract_c_positions` / `write_c_records` — methylation-specific
  fast paths used by `allc2cz` and `bam_to_cz`.

If the extension is not built, callers transparently fall back to
pure-Python implementations.

### `bam.py` — BAM → `.cz`

`bam_to_cz(bam_path, genome, output, mode, ...)` is the entry point.
It runs `samtools mpileup -B -f <fa> <bam>` as a subprocess and parses
each line into per-context (mc, cov) counts.

Three output **modes**:

| Mode | Stored columns | Use case |
| --- | --- | --- |
| `full` | `pos, strand, context, mc, cov` | Drop-in `allc.tsv` replacement |
| `pos_mc_cov` | `pos, mc, cov` | Position retained, context discarded |
| `mc_cov` | `mc, cov` | Aligned 1-to-1 with a reference `.cz`; missing sites filled with `(0, 0)`. Smallest output. |

The reverse-complement of the context window for G sites is computed
with `seq[lo:hi].translate(_RC_TABLE)[::-1]` — a C-level path that is
~30× faster than `"".join(_COMPLEMENT[b] for b in reversed(...))`.

`_LazyRefPositions` wraps the reference reader so that, for `mode='mc_cov'`,
positions for one chromosome can be streamed block by block instead of
materialising the full uint32 array (~316 MB for mm10 chr1). It calls
`reader.advise_sequential()` once and `reader.release_chunk(dim)` after
each chrom to keep RSS flat.

### `allc.py` — allc table I/O

* `allc2cz(allc_path, output, ...)` — read an `allc.tsv(.gz)` and write
  a `.cz`. Has a numpy fast path that batches per-chrom records before
  calling the Cython packer; same `advise_sequential` /
  `release_chunk` discipline as `bam.py`.
* `cz2allc(cz_path, output, ...)` — inverse direction.

### `dmr.py` / `dmr_accel.pyx` — DMR calling

* `call_dmr` (CG) builds, per reference chunk, `(n_cells, n_sites)` count
  matrices for both groups (workers cache their `Reader`s; chunks are
  submitted largest-first to a spawn pool) and runs the permutation RMS
  test in `dmr_accel.rms_run_sites` (Cython, OpenMP `prange`, per-site
  RNG seeded from `(seed, site)` so results are thread-count independent).
  1-vs-1 pseudobulk comparisons use a vectorised, SciPy-identical Fisher
  exact test with table de-duplication.
* `call_dmr_ch` pools counts into bins and normalises each cell's global
  mCH rate first; `call_dmr_one_vs_rest` runs a whole class in one pass
  (`_process_chunk_ovr`).
* `_merge_dms_to_dmr`, `annot_dmr`, `merge_dmr_results`, `call_dmr_array`.

Pseudo-reads / bedGraph export now live in `peaks.py`
(`call_peaks`, `call_peaks_bdg`, `to_bedgraph`).

### `features.py` — AnnData aggregation

`cz_to_anndata(...)` aggregates one or many `.cz` files over a feature
table (genes, fixed-width bins, peaks, …). Outputs an AnnData with
`mc` and `cov` layers. The chrom axis of the feature index is auto-
detected. For multi-file runs the aggregation is parallelised through
a multiprocessing pool whose workers share the feature index via
`_pool_init`.

Memory model (per worker ≈ one decompressed cell chunk, independent of
the genome / cohort size):

* For reference-aligned cells (no `pos` column) the **parent** resolves
  a *site → feature map* once (`_build_site_feature_map`): per chrom, run
  boundaries / per-feature run ranges plus either the context-index rows
  to gather (sparse selections, e.g. CG ≈ 4%) or the small excluded
  complement (dense selections, e.g. CH ≈ 96%: sum everything, subtract
  the complement). The index is streamed chrom by chrom (`_LazyIndexIds`),
  never materialised whole. Workers only sum mc / cov over the runs
  (`_run_sums` → Cython `cz_accel.c_run_sums`, one pass over the packed
  record bytes; numpy slab-wise `reduceat` fallback) and read features
  off a cumulative sum. Workers never open the reference, so there is no
  per-worker genome-wide position cache.
* The pool is created with the `fork` context (`_make_pool`) so the map /
  index dict are inherited copy-on-write; tasks are submitted through a
  bounded window (`_bounded_map`) so finished-but-unconsumed results
  cannot pile up; `jobs` is capped to the memory budget (`_cap_workers`,
  free RAM / cgroup headroom or `max_mem=`).
* **Fork + OpenMP:** libgomp is not fork-safe — a child forked from a
  process whose OpenMP thread pool is already live deadlocks on its first
  multi-threaded parallel region. `cytozip.cz` registers an
  `os.register_at_fork(after_in_child=...)` hook
  (`_pin_openmp_in_forked_child`) that caps the OpenMP threads of every
  Cython kernel (`cz_accel` inflate / deflate, `dmr_accel` permutation
  test) to `CYTOZIP_FORK_THREADS` (default 1) in **every** forked child, so
  all fork-based pools in the package (features, allc2cz, index, merge,
  pivot, peaks, dmr comb-p) are covered without per-pool code; the
  `features.py` pool initializers additionally pin explicitly (also under
  `spawn`). The cap also stops `jobs` workers from running `4 × jobs`
  threads. `dmr.py`'s main pools use the `spawn` context and are
  unaffected either way.
* Workers return only the covered `(col, mc, cov)` triplets
  (`_StreamingSparseBuilder.append_sparse`).
* The finalize stage (`_compute_beta_params_sparse`,
  `_compute_score_matrix_sparse`, `_compute_hvf_var_stats_sparse`,
  `_apply_nan_policy`) iterates over row blocks of ~`_BLOCK_NNZ` stored
  entries, so float64 temporaries do not scale with `nnz`; under
  `nan_policy='auto'` the dense fill is skipped when `n_cells × n_feat × 4 B`
  exceeds the budget.
* `verbose=True` (`--verbose`) shows progress bars (tqdm when installed)
  and logs wall time plus the sampled peak memory (process tree PSS +
  cgroup usage).

`make_genome_bins` produces a fixed-width bin BED-like dataframe
(used as the feature input for whole-genome 5kb / 100kb runs).

`cz_to_anndata_sites(...)` builds the **single-cytosine** `(n_cells,
n_sites)` matrix: each cell's chunk is inflated once (or block-gathered
via `Reader.gather_rows` for very sparse indexes), restricted to the
context index, and only covered sites are appended to the streaming CSR
builder (`_StreamingSparseBuilder.append_sparse`). `.X` is the raw or
per-cell posterior fraction (see `docs/Methods.md` §8).

### `merge.py` — merge `.cz` files

Streams multiple `.cz` files into a single output, preserving chunk
keys and column layout. Used to combine per-cell or per-chunk outputs.

### `__init__.py` — CLI

`czip <subcommand>` is implemented here. Each subcommand is a thin
argparse wrapper around one of the modules above
(`bam_to_cz`, `allc2cz`, `build_ref`, `index`, `merge_cz`, `pivot_*`,
`cz_to_anndata`, `cz_to_anndata_sites`, `call_dmr*`, `call_peaks*`,
`predict_cell_type`, `deconvolve_bulk`, …); heavy modules are imported
inside the branch that needs them so `czip --help` stays fast.

## Performance / memory patterns

A few patterns appear repeatedly in the codebase; following them keeps
RSS predictable on whole-genome runs.

1. **Slim copies from structured arrays.** Slicing a column from a
   numpy structured-array view (`arr[blk]['pos']`) returns a *view*
   that pins the parent buffer. Always materialise with
   `.astype(np.uint32, copy=True)` if you intend to release the source
   block.
2. **mmap discipline.** For any whole-file walk:
   ```python
   reader.advise_sequential()
   for dim in chroms:
       ...consume one chrom...
       reader.release_chunk(dim)
   ```
   This is the only reliable way to keep the kernel from pinning the
   entire reference file in your RSS — `MALLOC_TRIM_THRESHOLD_`,
   `MALLOC_ARENA_MAX`, jemalloc and tcmalloc do **not** help, because
   mmap'd file pages are not allocator-managed.
3. **Packed buffers for per-record state.** Per-chrom buffers in
   `bam_to_cz` use `array.array('I' / 'H')` rather than Python lists
   (~36 B per int object → 4 / 2 B per element).
4. **`malloc_trim(0)` after big drops.** After releasing a chrom's
   per-chrom buffers + ref slice, `bam.py` calls `libc.malloc_trim(0)`
   so glibc returns free'd top-of-heap pages to the kernel.

## Building the docs locally

```bash
cd docs
make html
# open docs/build/html/index.html
```

`recommonmark` is enabled so this file (Markdown) and the existing
`.rst` files build into the same Sphinx site. The HTML output is
checked in under `docs/build/html` and surfaced via GitHub Pages from
the `docs/` folder.
