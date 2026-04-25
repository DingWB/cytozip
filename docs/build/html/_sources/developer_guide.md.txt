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
  compressed *blocks* of at most 65535 raw bytes (raw DEFLATE).
* **Chunk tail** — appended after the last block of a chunk; contains
  the per-block virtual offsets and chunk-key values.
* **Chunk index** — appended at end-of-file before the EOF marker so HTTP
  / remote readers can locate any chunk in O(1) without scanning the
  whole file.
* **EOF marker** — 28-byte sentinel borrowed from BGZF.

A *virtual offset* packs `(block_start << 16) | within_block_offset` into
a single 64-bit integer, providing O(1) random access to any record.

Strictly-monotonic integer columns (typically genomic positions) can be
stored with **delta encoding** (`_ENC_DELTA`): within each block we store
first-order differences, so DEFLATE compresses a `pos` column to a
fraction of its raw size. Decoding is a single numpy `cumsum` per
delta column per block.

## Module map

| Module | Lines | Purpose |
| --- | --- | --- |
| `cytozip/cz.py` | ~3800 | Core `Reader` / `Writer` and the on-disk format |
| `cytozip/cz_accel.pyx` | — | Cython accelerators (libdeflate-backed block (de)compression, record packing, chunk index parser) |
| `cytozip/bam.py` | ~720 | Pileup-based BAM → `.cz` methylation extraction |
| `cytozip/allc.py` | ~520 | `allc.tsv(.gz)` ↔ `.cz` conversion |
| `cytozip/dmr.py` | ~600 | DMR utilities: pseudo-reads BED, bedGraph export |
| `cytozip/features.py` | ~1100 | Aggregation over genomic features → AnnData (gene/bin/peak by cell matrices) |
| `cytozip/merge.py` | ~430 | Merge multiple `.cz` files into one |
| `cytozip/__init__.py` | ~590 | CLI entry point (`czip ...`) |

### `cz.py` — core format

* **`Writer`** — open, accumulate records into the active block, flush
  full blocks (`_BLOCK_MAX_LEN = 65535` raw bytes), close out the chunk
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

### `dmr.py` — pseudo-reads + bedGraph

* `make_pseudo_reads_bed` — turns a `.cz` into the per-cell pseudo-reads
  BED format consumed by methylpy.
* `to_bedgraph` — collapses a `.cz` into a bedGraph track for genome
  browsers.

Both walk per-chromosome blocks against a reference `.cz`; both call
`advise_sequential()` + `release_chunk(dim)` after each chrom.

### `features.py` — AnnData aggregation

`cz_to_anndata(...)` aggregates one or many `.cz` files over a feature
table (genes, fixed-width bins, peaks, …). Outputs an AnnData with
`mc` and `cov` layers. The chrom axis of the feature index is auto-
detected. For multi-file runs the aggregation is parallelised through
a multiprocessing pool whose workers share the feature index via
`_pool_init`.

`make_genome_bins` produces a fixed-width bin BED-like dataframe
(used as the feature input for whole-genome 5kb / 100kb runs).

### `merge.py` — merge `.cz` files

Streams multiple `.cz` files into a single output, preserving chunk
keys and column layout. Used to combine per-cell or per-chunk outputs.

### `__init__.py` — CLI

`czip <subcommand>` is implemented here. Each subcommand is a thin
argparse wrapper around one of the modules above
(`bam_to_cz`, `allc2cz`, `cz2allc`, `header`, `summary`, `merge`,
`features`, `bedgraph`, …).

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
