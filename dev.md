# cytozip (.cz) Binary File Format Specification

## File Layout Overview
```
┌──────────────────────────────────────────────┐
│ FILE HEADER (variable size, ~60 bytes)       │
│  Magic(4B) Version(4B) TotalSize(8B)         │
│  Message + Formats + Columns                 │
│  + SortCol(1B) + Dimensions                  │
├──────────────────────────────────────────────┤
│ CHUNK #1 (e.g. chr1)                         │
│  ┌ Chunk Header: "CC"(2B) + ChunkSize(8B)   │
│  ├ BLOCK #1: "CB"(2B) + BSize(2B)           │
│  │   + compressed_data + RawLen(2B)          │
│  ├ BLOCK #2 ...                              │
│  └ CHUNK TAIL:                               │
│     DataLen(8B) + NBlocks(8B)                │
│     + VirtualOffsets(N×8B)                   │
│     + FirstCoords(N × sort_col fmt, opt.)    │
│     + DimValues                              │
├──────────────────────────────────────────────┤
│ CHUNK #2 (e.g. chr2) ...                     │
├──────────────────────────────────────────────┤
│ CHUNK INDEX (for O(1) chunk lookup)          │
│  "CZIX"(4B) + NChunks(8B)                   │
│  For each chunk:                             │
│    DimValues + Start + Size + DataLen        │
│    + NBlocks                                 │
├──────────────────────────────────────────────┤
│ ChunkIndexOffset (8B)                        │
│ EOF marker (28B, BGZF compatible)            │
└──────────────────────────────────────────────┘
```

## Format & size
https://docs.python.org/3/library/struct.html#format-characters

All multi-byte fields in .cz use **little-endian** byte order (prefix `<` in Python struct format strings).
Byte order determines how multi-byte values are stored in memory:
- **Little-endian (`<`)**: least-significant byte first. E.g. the 32-bit integer `0x01020304` is stored as bytes `04 03 02 01`.
- **Big-endian (`>`)**: most-significant byte first. The same integer is stored as `01 02 03 04`.

Little-endian is the native byte order on x86/x86-64 and ARM (the vast majority of modern CPUs),
so using `<` avoids byte-swapping overhead on these architectures.

| Format | C Type             | Python type       | Standard size | Max value / range               |
| ------ | ------------------ | ----------------- | ------------- | ------------------------------- |
| `x`    | pad byte           | no value          |               | —                               |
| `c`    | char               | bytes of length 1 | 1             | 1 char                          |
| `b`    | signed char        | integer           | 1             | -128 to 127                     |
| `B`    | unsigned char      | integer           | 1             | 0 to 255                        |
| `?`    | _Bool              | bool              | 1             | True / False                    |
| `h`    | short              | integer           | 2             | -32,768 to 32,767               |
| `H`    | unsigned short     | integer           | 2             | 0 to 65,535                     |
| `i`    | int                | integer           | 4             | -2,147,483,648 to 2,147,483,647 |
| `I`    | unsigned int       | integer           | 4             | 0 to 4,294,967,295              |
| `l`    | long               | integer           | 4             | -2,147,483,648 to 2,147,483,647 |
| `L`    | unsigned long      | integer           | 4             | 0 to 4,294,967,295              |
| `q`    | long long          | integer           | 8             | ±9.2×10^18                      |
| `Q`    | unsigned long long | integer           | 8             | 0 to 1.8×10^19                  |
| `e`    | _Float16           | float             | 2             | ±65,504                         |
| `f`    | float              | float             | 4             | ±3.4×10^38                      |
| `d`    | double             | float             | 8             | ±1.8×10^308                     |
| `s`    | char[]             | bytes             |               | N bytes (use `Ns`, e.g. `3s`)   |
| `p`    | char[]             | bytes             |               | up to 255 chars                 |

## Header Structure
| Offset | Field | Type | Size | Description |
|--------|-------|------|------|-------------|
| 0 | magic | 4s | 4B | `CZIP` |
| 4 | version | `<f` | 4B | 0.1 |
| 8 | total_size | `<Q` | 8B | File size excluding chunk_index + EOF |
| 16 | msg_len | `<H` | 2B | Message string length |
| 18 | message | s | var | UTF-8 message (e.g. genome assembly) |
| var | n_cols | `<B` | 1B | Number of columns |
| var | formats[] | B+s | var | Per column: len(1B) + format string |
| var | columns[] | B+s | var | Per column: len(1B) + column name |
| var | sort_col | `<B` | 1B | Index of sort column, or `0xFF` (255) if none |
| var | n_chunk_dims | `<B` | 1B | Number of chunk_dims |
| var | dims[] | B+s | var | Per dim: len(1B) + dim name |

`sort_col` identifies a single integer column whose values are monotonically
non-decreasing within every chunk (e.g. `pos` for allc). When enabled, each
chunk tail stores the first record's `sort_col` value for every block,
enabling true O(log N) bisect on numeric coordinates without decompressing
probe blocks. `0xFF` disables the feature; readers then fall back to
decompressing the first record of candidate blocks during `query`.

## Chunk Header

| Field | Type | Size | Description |
|-------|------|------|-------------|
| magic | 2s | 2B | `CC` |
| chunk_size | `<Q` | 8B | Byte size from chunk start to chunk tail (excludes tail) |

## Block Structure (6B overhead per block)

| Field | Type | Size | Description |
|-------|------|------|-------------|
| magic | 2s | 2B | `CB` |
| block_size | `<H` | 2B | compressed_data + 6 |
| compressed_data | bytes | var | Raw DEFLATE (-15 wbits) |
| raw_len | `<H` | 2B | Uncompressed data length |

## Chunk Tail Structure

| Field | Type | Size | Description |
|-------|------|------|-------------|
| data_len | `<Q` | 8B | Total uncompressed bytes |
| n_blocks | `<Q` | 8B | Number of blocks |
| virtual_offsets[] | `<Q`×N | 8B×N | `(block_disk_offset << 16) \| within_block_offset` |
| first_coords[] | fmt×N | k×N | First record's `sort_col` value per block (only if `sort_col != 0xFF`; `k` = size of `sort_col`'s format) |
| chunk_key_values[] | B+s | var | Dimension value strings |

## Chunk Index (end of file, for remote/partial reading)
```
"CZIX" (4B magic)
n_chunks (Q, 8B)
For each chunk:
  [dim_len(B) + dim_value(s)] × n_chunk_dims
  chunk_start_offset (Q, 8B)
  chunk_size (Q, 8B)
  chunk_data_len (Q, 8B)
  chunk_nblocks (Q, 8B)
```

Block virtual offsets are stored only in each chunk's tail (not duplicated here)
and are read on demand via `_load_chunk()`.

**Remote reading workflow (3 HTTP Range requests):**
1. `bytes=0-200` → parse header
2. `bytes=(size-36)-(size-1)` → read `chunk_index_offset(8B) + EOF(28B)`
3. `bytes=idx_offset-(size-37)` → read chunk index → O(1) jump to any chunk/block

---

## Compression

Blocks are compressed with raw DEFLATE (`-15` wbits, no zlib/gzip wrapper).
The native reader/writer links against **[libdeflate](https://github.com/ebiggers/libdeflate)**
(via Cython in `cz_accel.pyx`), which is 2–3× faster than zlib for both
compress and decompress while producing fully compatible output. Pure-Python
fallbacks and the browser reader use standard DEFLATE decoders (`zlib.decompress(-15)`
and `DecompressionStream('deflate-raw')` respectively), so files remain
interoperable.

Build requirements:
- `libdeflate.so` and `libdeflate.h` available (e.g. `conda install -c conda-forge libdeflate`)
- `setup.py` links with `-ldeflate` from `$CONDA_PREFIX`

## Installation
```shell
python setup.py build_ext --inplace 
# or
pip install -e .
# install from local disk
pip uninstall -y cytozip && python3 -m pip install .
# rebuild .pyx
python setup.py build_ext --inplace
python -c "import cytozip.cz_accel; print(cytozip.cz_accel.__file__)"
python -c "from cytozip.cz import Reader"
```

## Reference file
```shell
time czip build_ref -g ~/Ref/mm10/mm10_ucsc_with_chrL.fa -O ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -t 20
# 0m44.284s

# create a coordinate index for all CG (including forward and reverse strand)
time czip index context -I ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -p CGN -O ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz.CGN.idx.cz
# 8m24.544s

# create a coordinate index for all CG (forward strand only)
time czip index context -I ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -p +CGN -O ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz.CGN.forward.idx.cz
# about 5 minutes

# use the forward CG index to extract forward strand CG coordinates from reference
time czip extract -i ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -s ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz.CGN.forward.idx.cz -o ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz
# about 1m23.855s

# Index files are themselves .cz files — inspect with `czip view -I *.idx.cz --show-keys 0`
```

## Remote Reading
Read .cz files from a remote HTTP/HTTPS server using HTTP Range requests.
Requires the file to have a chunk index.
Initialization takes 2-3 HTTP requests (header + chunk index); each chunk fetch takes ~1 request per 2MB of compressed data.

### Python API
```python
from cytozip.cz import Reader

# Auto-detect URL
r = Reader("https://server.com/data/mm10_ref.allc.cz")

# Or explicit factory with custom cache size (default 2MB)
r = Reader.from_url("https://server.com/data/mm10_ref.allc.cz", cache_size=4*1024*1024)

# Print header
r.print_header()

# List chunks (chromosomes)
print(r.chunk_info)

# Fetch all records for one chromosome
for record in r.fetch(("chr1",)):
    print(record)

# Fetch raw bytes (for numpy processing)
import numpy as np
raw = r.fetch_chunk_bytes(("chr1",))
dt = np.dtype([("pos", "<u8"), ("mc", "<u2"), ("cov", "<u2")])
arr = np.frombuffer(raw, dtype=dt)

# Query by region
results = list(r.query(chunk_key="chr9", start=3000294, end=3000472, printout=False))

# Read chunk index directly
idx = r.read_chunk_index()
print(idx[("chr1",)])  # {'start': ..., 'size': ..., 'data_len': ..., 'nblocks': ..., 'block_vos': [...]}

r.close()
```

### How it works
`RemoteFile` wraps HTTP Range requests into a file-like object (`read`/`seek`/`tell`/`close`) with a 2MB read-ahead cache. This is transparent to Reader — all methods (fetch, query, view, subset, etc.) work identically on local and remote files.

An optional `session` parameter (`requests.Session`) can be passed to `RemoteFile` or `Reader.from_url()` for servers that require cookies or special authentication (e.g. Figshare behind WAF).

```
Init (2-3 HTTP requests):
  1. HEAD → get file size
     (fallback: Range GET bytes=0-0 probe if HEAD returns 202/WAF challenge
      or omits Content-Length — parses Content-Range header instead)
  2. GET Range bytes=0-2MB → parse header (cached, also covers first chunks)
  3. GET Range bytes=(size-2MB)-(size-1) → read chunk_index_offset + chunk index + EOF

Per-chunk fetch (1+ requests):
  GET Range bytes=chunk_start-(chunk_start+2MB) → decompress blocks → yield records
```

### Figshare example
Figshare uses CloudFront WAF which requires browser-like headers and cookies:
```python
import requests
from cytozip.cz import Reader

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 ...",
    "Referer": "https://figshare.com/",
    "Accept": "*/*",
})
session.get("https://figshare.com")  # acquire cookies
reader = Reader.from_url("https://figshare.com/ndownloader/files/XXXXX", session=session)
```

### JavaScript API
Read remote .cz files from the browser using `cz_reader.mjs` (ES module).
Uses `fetch()` + HTTP Range requests + `DecompressionStream` (no server-side dependencies).

#### Browser usage
```html
<script type="module">
import { CzReader } from './cytozip/cz_reader.mjs';

const reader = await CzReader.fromUrl(
  'https://figshare.com/ndownloader/files/63531984',
  { fetchOptions: { credentials: 'include' } }
);

// Inspect header
console.log(reader.header);
// { magic: 'CZIP', formats: ['Q','B','B'], columns: ['pos','mc','cov'],
//   sortCol: 0, chunk_dims: ['chrom'], ... }

// List all chunks (chromosomes)
console.log(reader.chunkKeys);
// ['chr1', 'chr2', ..., 'chrY']

// Summary
console.table(reader.summaryChunks());

// Fetch all records for one chromosome
const records = await reader.fetch('chr9');
console.log(`chr9: ${records.length} records`);
console.log(records[0]); // [pos, mc, cov]

// Query a genomic region (binary search, O(log N) block decompressions)
const hits = await reader.query('chr9', 60610139, 60610151);
console.log(`Query returned ${hits.length} records:`);
hits.forEach(r => console.log(r));

// Raw bytes for typed-array processing (fastest)
const raw = await reader.fetchChunkBytes('chr9');
const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
// Parse with DataView or create a structured typed array

reader.close();
</script>
```

#### Node.js usage (v18+)
```javascript
import { CzReader } from './cytozip/cz_reader.mjs';

const reader = await CzReader.fromUrl(
  'https://figshare.com/ndownloader/files/63531984'
);
console.log(reader.header);
const results = await reader.query('chr9', 60610139, 60610151);
results.forEach(r => console.log(r));
reader.close();
```

#### CORS note
When reading from Figshare or other third-party servers directly in the browser,
you may hit CORS restrictions. Solutions:
1. Host the .cz file on a CORS-enabled server/CDN (e.g. S3 with proper headers).
2. Use a lightweight proxy (e.g. `cors-anywhere` or a Cloudflare Worker).
3. For local development, use a local HTTP server serving the .cz file.

## docs
```shell
pip install sphinx sphinx-autobuild sphinx-rtd-theme pandoc nbsphinx sphinx_pdj_theme sphinx_sizzle_theme recommonmark readthedocs-sphinx-search
conda install conda-forge::pandoc

mkdir -p docs && cd docs
sphinx-quickstart
# Separate source and build directories (y/n) [n]: y
# Project name: adataviz

# vim source/conf.py
# add *.rst

# cd docs
# vim index.html: <meta http-equiv="refresh" content="0; url=./build/html/index.html" />
cd docs
rm -rf build
ln -s ~/Projects/Github/cytozip/notebooks source/notebooks
sphinx-apidoc -e -o source -f ../../cytozip
make html
rm -rf source/notebooks
cd ..
ls
ls docs

vim .nojekyll #create empty file
```

## Run test & notebooks
```shell
cd /home/x-wding2/Projects/Github/cytozip
rm -rf cytozip_example_data/output/cz
rm -rf cytozip_example_data/output/allc
python tests/benchmark_bam_to_cz.py  -j 20
python tests/benchmark_allc_to_cz.py -j 20
python tests/benchmark_query.py
nbexe notebooks/2.dnam.ipynb
```

## Package Rename Candidates

The name "cytozip" is too narrow — the package is not just a compression tool but a full single-cell DNA methylation analysis framework (DMR, clustering, motif analysis, etc.). Candidate names for publication:

| Name | Meaning | Pros |
|------|---------|------|
| **mCyte** | **m**(ethyl) + **Cyte**(cytosine/cell) | Short, memorable, "m" prefix well-known in epigenetics (5mC) |
| **EpiCyte** | **Epi**(genetic) + **Cyte**(cytosine/cell) | Broader scope, extensible to other epigenomic analyses |
| **CytoMine** | **Cyto**(sine) + **Mine**(data mining) | Emphasizes mining insights from massive sc-methylation data |

Top recommendations: **mCyte** (concise, domain-specific) or **MESA**.
---

## Roadmap: Submission Plan
Strategic to-do list for scaling cytozip into a publication-ready framework supporting single-cell DNA methylation + methylation array data, with an online portal and expanded analysis features.

### 1. Paper Positioning

- [ ] Decide main narrative (recommended: *"A unified, cloud-native data format and analysis ecosystem for population-scale methylomes"*).
- [ ] Benchmark vs. ALLCools / methylpy / bsseq / tabix+bgzip / TileDB / Zarr / Parquet on four axes: compression ratio, random-access latency, region query, cross-sample join.
- [ ] Prepare at least one flagship biological application showcasing interactive-speed analysis over millions of cells.
- [ ] Unify sc-methylation + Illumina array (450K/EPIC/EPICv2) under the same format — strong differentiator.

### 2. Format / Algorithm Layer

- [ ] Freeze **CZ format spec v1.0** — publish `docs/spec.md` formalizing the existing `CZIP` magic (4B) + version float (4B) already in the header. Define a policy to actually *use* the version field for backward-compatible reads on future header changes (prior DELTA-encoding addition broke old files — don't repeat).
- [ ] Add array-data payload: dense matrix block + per-column zstd (TileDB-style tiles).
- [ ] Implement importers: `idat2cz`, `sesame2cz`, `minfi2cz`.
- [ ] Extend existing `catcz` output (already: many cells → one `.cz` with shared header + per-cell chunk + `chunk_key2offset` index) with two additions for cohort-scale use:
    - Embed cell-level **obs metadata** (Arrow IPC / Parquet footer) so `reader.obs` returns a `pandas.DataFrame` — no external `obs.tsv` to keep in sync.
    - Build a **group inverted index** (e.g. `cluster → [cell_id, ...]`) to coalesce remote range reads for queries like "all Oligo cells at chr9:60610139-60610151".
- [ ] Add codec variants: RLE (long methylated runs), FOR (frame-of-reference for mc/cov), cross-block zstd dict for reference-aligned data.
- [ ] Benchmark each codec's gain in the paper table.
- [ ] Export C header + shared library from Cython core for R / Julia / Rust bindings.

### 3. Analysis Features

- [ ] `call_dmr`: port DSS / DMRfind to run natively on `.cz` streams (zero decompress to numpy) — target ≥10× speedup vs. methylpy/ALLCools.
- [ ] `call_peak`: sliding-window + Poisson enrichment for 5hmC / mCH, input `.cz`.
- [ ] Clustering: do NOT rewrite; export `AnnData` / `MuData` compatible objects for scanpy / ALLCools. Sell the I/O + feature extraction speed.
- [ ] Unified API:
  - `cz.pileup(region) -> array`
  - `cz.call_dmr(groups, region)`
  - `cz.to_anndata(features="100kb" | "gene" | "peaks")`
- [ ] Provide Snakemake / Nextflow wrappers.

### 4. Portal & Ecosystem

- [ ] Backend: FastAPI + `.cz` HTTP-range server — stream directly from S3/GCS without full download.
- [ ] Frontend: React + igv.js / higlass for tracks (no custom genome browser).
- [ ] Flagship datasets: BICAN / HMBA / mouse_dev cohorts as paper Figure 4 material.
- [ ] `cz.open("s3://bucket/file.cz")` and `https://...` support, fetching only queried blocks.
- [ ] `czserve` CLI for local portal deployment.
- [ ] PyPI + bioconda release; cibuildwheel prebuilt wheels for linux/mac/arm64.
- [ ] Thin R package `cytozipR` (reticulate or C-API bindings).

### 5. Engineering / Pre-submission Checklist

- [ ] Versioned format spec document (`docs/spec.md`).
- [ ] Bump version field + add version-aware read branches before any future header change (magic + version bytes already present).
- [ ] Unit-test coverage >80%, CI on linux/mac/windows.
- [ ] Reproducible benchmark suite under `benchmarks/`, archived with Zenodo DOI.
- [ ] Docs on readthedocs with tutorials.
- [ ] Colab-runnable demo notebooks.
- [ ] Docker image.
- [ ] bioRxiv preprint before NM submission.

### 6. High-Level Timeline

1. Freeze spec v1.0 (document existing magic + version fields, define version-bump policy) — ~2 weeks.
2. Array support + obs/group-index extension to catcz output — ~1–2 months.
3. Native `call_dmr` + benchmark — ~1 month.
4. Portal MVP (FastAPI + S3 range read + one flagship dataset) — ~1–2 months.
5. bioRxiv → Nature Methods submission.

---

## New modules (this iteration)

### `cytozip/bam.py` — `bam_to_cz`
Port of ALLCools `_bam_to_allc.bam_to_allc` (original author: Yupeng He) that writes
directly into `.cz` instead of ALLC `tsv.gz` + tabix. Single pass over
`samtools mpileup` output; emits `(pos, strand, context, mc, cov)` records
(or `(pos, mc, cov)` when `--slim` is used). `pos` column is DELTA-encoded;
`sort_col='pos'` gives O(log N) region queries from CLI.

CLI: `czip bam_to_cz -I sample.bam -r ref.fa -O sample.cz [--slim] [--convert-strandness]`

### `cytozip/features.py` — `cz_to_anndata` / `parse_features`
Build a cell × feature `AnnData` over a BED / BED.gz / BED.bgz feature set.
Inputs may be:
1. A list of single-cell `.cz` files,
2. A directory of `.cz` files,
3. One `catcz`-merged `.cz` with `chunk_dims=[cell_id, chrom]` (the cell id
   chunk_key prefix is auto-detected).

Features grouped by chrom for I/O locality; per-region aggregation uses
`Reader.query()` which engages the Cython `c_query_regions` path. Output
has `X = mc/cov` (float32) plus integer CSR layers `mc` and `cov`.

CLI: `czip cz_to_anndata -I cell*.cz -f gene_2kb.bed.bgz -O out.h5ad`

## Rename: `chunksize` → `batch_size`
Post-`chunk_dims` rename, the term "chunk" now refers exclusively to the
on-disk file structure. The old `chunksize` parameter (rows flushed per
write) was overloaded and is now `batch_size` everywhere. The on-disk field
`chunk_size` (byte size of a chunk) is unchanged.

Affected:
- Python: `Writer.tocz`, `Writer.catcz`, `allc2cz`, `WriteC`, `extractCG`,
  `merge_cz`, `bam_to_cz`, cz_accel.c_write_c_records.
- CLI: `-c / --batch-size` (was `--chunksize`).

## Update: `bam_to_cz` storage modes + `cz_to_anndata` numpy fast path

### `bam_to_cz(mode=...)`
Three on-disk layouts selectable via ``mode``:

| mode | columns | size / site | notes |
|---|---|---|---|
| ``full`` (default) | ``[pos, strand, context, mc, cov]`` | ~13 B post-DEFLATE | self-contained |
| ``pos_mc_cov``     | ``[pos, mc, cov]``                  | ~5 B  post-DEFLATE | needs ref for context |
| ``mc_cov``         | ``[mc, cov]``                       | ~2-4 B post-DEFLATE | **requires ``reference``**; output positions are aligned 1:1 against the reference (missing sites filled with ``(0, 0)``) |

The legacy ``--slim`` flag is an alias for ``--mode pos_mc_cov``.

### `cz_to_anndata` numpy fast path
The per-region aggregation loop now uses
``Reader.fetch_chunk_bytes`` + ``np.frombuffer`` + ``np.searchsorted`` on
cumulative mc/cov arrays (same pattern as ``allc2cz``'s vectorised
reference-alignment branch). For typical cell × feature matrices this
is ~50-100x faster than the previous ``Reader.query`` Python loop.

`cz_to_anndata` also gained a ``reference=`` parameter that provides
positions for ``mc_cov``-only cells. The chrom axis of the dim tuple is
auto-detected (robust to catcz ordering).
