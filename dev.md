# cytozip (.cz) Binary File Format Specification

## File Layout Overview
```
┌──────────────────────────────────────────────┐
│ FILE HEADER (variable size, ~60 bytes)       │
│  Magic(4B) Version(4B) TotalSize(8B)         │
│  Message + Formats + Columns + Dimensions    │
├──────────────────────────────────────────────┤
│ CHUNK #1 (e.g. chr1)                         │
│  ┌ Chunk Header: "CC"(2B) + ChunkSize(8B)   │
│  ├ BLOCK #1: "CB"(2B) + BSize(2B)           │
│  │   + compressed_data + RawLen(2B)          │
│  ├ BLOCK #2 ...                              │
│  └ CHUNK TAIL:                               │
│     DataLen(8B) + NBlocks(8B)                │
│     + VirtualOffsets(N×8B)                   │
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
| var | n_dims | `<B` | 1B | Number of dimensions |
| var | dims[] | B+s | var | Per dim: len(1B) + dim name |

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
| dim_values[] | B+s | var | Dimension value strings |

## Chunk Index (end of file, for remote/partial reading)
```
"CZIX" (4B magic)
n_chunks (Q, 8B)
For each chunk:
  [dim_len(B) + dim_value(s)] × n_dims
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
time czip build_ref -g ~/Ref/mm10/mm10_ucsc_with_chrL.fa -O ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -n 20
# 0m44.284s

# create subset index for all CG (including forward and reverse strand)
time czip generate_ssi1 ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -p CGN -o ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz.CGN.ssi
# 8m24.544s

# create subset index for all CG (forward strand only)
time czip generate_ssi1 ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -p +CGN -o ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz.CGN.forward.ssi
# about 5 minutes

# using forward CG subset index to extract forward strand CG coordinates from reference
time czip extract -i ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -s ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz.CGN.forward.ssi -o ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz
# about 1m23.855s

# Actually, *.ssi is also a czip file, we can view .ssi using `czip view -I *.ssi --show-dim 0`
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
results = list(r.query(Dimension="chr9", start=3000294, end=3000472, printout=False))

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
//   dimensions: ['chrom'], ... }

// List all chunks (chromosomes)
console.log(reader.dims);
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