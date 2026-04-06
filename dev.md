# cytozip (.cz) Binary File Format Specification

## File Layout Overview
```
┌──────────────────────────────────────────────┐
│ FILE HEADER (variable size, ~60 bytes)       │
│  Magic(5B) Version(4B) TotalSize(8B)         │
│  Message + Formats + Columns + Dimensions    │
│  DeltaCols                                   │
├──────────────────────────────────────────────┤
│ CHUNK #1 (e.g. chr1)                         │
│  ┌ Chunk Header: "MC"(2B) + ChunkSize(8B)   │
│  ├ BLOCK #1: "MB"(2B) + BSize(2B)           │
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
│    DimValues + Offsets + BlockVirtualOffsets  │
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
| 0 | magic | 5s | 5B | `BMZIP` |
| 5 | version | `<f` | 4B | 0.1 |
| 9 | total_size | `<Q` | 8B | File size excluding chunk_index + EOF |
| 17 | msg_len | `<H` | 2B | Message string length |
| 19 | message | s | var | UTF-8 message (e.g. genome assembly) |
| var | n_cols | `<B` | 1B | Number of columns |
| var | formats[] | B+s | var | Per column: len(1B) + format string |
| var | columns[] | B+s | var | Per column: len(1B) + column name |
| var | n_dims | `<B` | 1B | Number of dimensions |
| var | dims[] | B+s | var | Per dim: len(1B) + dim name |
| var | n_delta | `<B` | 1B | Number of delta-encoded columns |
| var | delta_cols[] | B | var | Column indices with delta encoding |

## Block Structure (6B overhead per block)

| Field | Type | Size | Description |
|-------|------|------|-------------|
| magic | 2s | 2B | `MB` |
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
  [block_virtual_offset (Q)] × n_blocks
```

**Remote reading workflow (3 HTTP Range requests):**
1. `bytes=0-200` → parse header
2. `bytes=(size-36)-(size-1)` → read `chunk_index_offset(8B) + EOF(28B)`
3. `bytes=idx_offset-(size-37)` → read chunk index → O(1) jump to any chunk/block

## Delta Encoding

For sorted coordinate columns, store differences instead of absolute values.
Each block's first record stores the absolute value; subsequent records store deltas.
Blocks are aligned to record boundaries (block_size is a multiple of unit_size).

```
Example (position column):
  Absolute:  3012489, 3012491, 3012530, 3012578   (8B each, high entropy)
  Delta:     3012489,       2,      39,      48   (compresses much better)
```

---

## Installation
```shell
python setup.py build_ext --inplace 
# or
pip install -e .
# install from local disk
pip uninstall -y cytozip && python3 -m pip install .
```

## allc to cz
```shell
time cytozip AllC -G ~/Ref/mm10/mm10_ucsc_with_chrL.fa -O ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -n 20 run

rm -f example/example_data/FC_E17a_3C_1-1-I3-F13.with_coordinate.cz
time cytozip bed2cz example/example_data/FC_E17a_3C_1-1-I3-F13.allc.tsv.gz example/example_data/FC_E17a_3C_1-1-I3-F13.with_coordinate.cz -F Q,H,H -C pos,mc,cov -u 1,4,5
# 2m5.719s -> 1m32.432s

rm -f example/example_data/FC_E17a_3C_1-1-I3-F13.cz
time cytozip bed2cz example/example_data/FC_E17a_3C_1-1-I3-F13.allc.tsv.gz example/example_data/FC_E17a_3C_1-1-I3-F13.cz -r ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz
# 8m40.090s -> 2m0.835s

cytozip Reader -I example/example_data/FC_E17a_3C_1-1-I3-F13.cz print_header
cytozip Reader -I example/example_data/FC_E17a_3C_1-1-I3-F13.cz view -s 0 |head

cytozip Reader -I ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz summary_chunks | head

cytozip Reader -I example/example_data/FC_E17a_3C_1-1-I3-F13.with_coordinate.cz print_header
cytozip Reader -I example/example_data/FC_E17a_3C_1-1-I3-F13.with_coordinate.cz view -s 0 |head

cytozip Reader -I example/example_data/FC_E17a_3C_1-1-I3-F13.with_coordinate.cz query -D chr9 -s 3000294 -e 3000472 | awk '$3>50'

```

## Query
```shell

tabix example/example_data/FC_E17a_3C_1-1-I3-F13.allc.tsv.gz chr9 | awk '$5 > 50' |head
cytozip Reader -I example/example_data/FC_E17a_3C_1-1-I3-F13.cz query -r ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -D chr9 -s 3000294 -e 3000472 |awk '$5>50'
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

```
Init (2-3 HTTP requests):
  1. HEAD → get file size
  2. GET Range bytes=0-2MB → parse header (cached, also covers first chunks)
  3. GET Range bytes=(size-2MB)-(size-1) → read chunk_index_offset + chunk index + EOF

Per-chunk fetch (1+ requests):
  GET Range bytes=chunk_start-(chunk_start+2MB) → decompress blocks → yield records
```

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
ln -s ~/Projects/Github/czip/notebooks source/notebooks
sphinx-apidoc -e -o source -f ../../cytozip
make html
rm -rf source/notebooks
cd ..
ls
ls docs

vim .nojekyll #create empty file
```