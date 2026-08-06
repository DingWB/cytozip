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

## Block Structure (10B overhead per block)

| Field | Type | Size | Description |
|-------|------|------|-------------|
| magic | 2s | 2B | `CB` |
| block_size | `<I` | 4B | compressed_data + 10 |
| compressed_data | bytes | var | Raw DEFLATE (-15 wbits) |
| raw_len | `<I` | 4B | Uncompressed data length |

Maximum uncompressed block size is 256 KiB − 1 (chosen empirically; DEFLATE's
32 KiB sliding window saturates compression at 256 KiB while keeping per-block
point queries ~4× faster than 1 MiB blocks).

## Chunk Tail Structure

| Field | Type | Size | Description |
|-------|------|------|-------------|
| data_len | `<Q` | 8B | Total uncompressed bytes |
| n_blocks | `<Q` | 8B | Number of blocks |
| virtual_offsets[] | `<Q`×N | 8B×N | `(block_disk_offset << 20) \| within_block_offset` (44+20 split) |
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
time czip build_ref -g ~/Ref/mm10/mm10_ucsc_with_chrL.fa -O ~/Ref/mm10/annotations/mm10_with_chrL.allc.cz -j 20
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

## Python API quick start

```python
import cytozip as cz, pandas as pd

# Write — infer struct formats from DataFrame dtypes, partition by 'chrom'
df = pd.DataFrame({'chrom': 'chr1', 'pos': [...], 'mc': [...], 'cov': [...]})
cz.Writer.from_dataframe(df, 'out.cz',
                         partition_by='chrom',
                         sort_col='pos',           # column name, not index
                         delta_cols=['pos'])

# Read — unified open() + Pythonic helpers
with cz.open('out.cz') as r:
    print(len(r))                          # total record count
    print(r.head(5))                       # first 5 records
    print(list(r.regions()))               # (chunk_key, min_pos, max_pos) per chunk
    df_chr1 = r.to_pandas(chunk_key=('chr1',))
    df_regs = r.to_pandas(regions=[(('chr1',), 1000, 2000),
                                    (('chr2',), 5000, 6000)])
    with r.region(('chr1',), 1000, 2000) as records:
        for rec in records:
            ...
    with r.region(('chr1',), 1000, 2000, as_numpy=True) as arr:
        ...                                # structured ndarray, vectorized
    for rec in r:                          # iterate every record
        ...
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

## Enabling CORS on the neomorph server

The neomorph server (`https://neomorph.salk.edu/ftp/bican/`) hosts `.cz` files
served by **Apache2 on CentOS/RHEL**. To let the browser reader (`cz_reader.mjs`
/ `cz_viewer.html`) fetch these files directly, the server must send CORS
headers **and** support HTTP Range requests. Both are enabled through
`mod_headers`.

### 1. Enable the required Apache module
```shell
# CentOS/RHEL: mod_headers ships with httpd; just make sure it's loaded.
# Add (or confirm) this line in /etc/httpd/conf/httpd.conf:
LoadModule headers_module modules/mod_headers.so

# Verify it is active:
httpd -M 2>/dev/null | grep headers
# expected: headers_module (shared)
```

### 2. Add the CORS + Range headers
Edit `/etc/httpd/conf/httpd.conf` (or drop a file in
`/etc/httpd/conf.d/cors.conf`). Scope the rules to the directory that serves
the `.cz` files so other paths are unaffected:
```apache
<Directory "/var/www/html/ftp/bican">
    # Allow any origin to read the files (use a specific origin to lock down)
    Header set Access-Control-Allow-Origin "*"
    Header set Access-Control-Allow-Methods "GET, HEAD, OPTIONS"
    Header set Access-Control-Allow-Headers "Range, Content-Type"

    # CRITICAL for cz remote reading: the browser must see these response
    # headers to perform Range requests and O(1) chunk lookups.
    Header set Access-Control-Expose-Headers "Content-Range, Accept-Ranges, Content-Length"
    Header set Accept-Ranges "bytes"

    # Answer CORS pre-flight (OPTIONS) requests with 204 instead of hitting a file
    RewriteEngine On
    RewriteCond %{REQUEST_METHOD} OPTIONS
    RewriteRule ^(.*)$ $1 [R=204,L]
</Directory>
```

> Notes
> - `Access-Control-Expose-Headers` is the header most often forgotten. Without
>   `Content-Range` / `Accept-Ranges` / `Content-Length` exposed, the browser
>   `fetch()` cannot read Range metadata and the reader fails even though the
>   bytes are returned.
> - Use `Header set` (not `Header add`) to avoid duplicate values on redirects.
> - Serve over **HTTPS** — the viewer page is HTTPS and mixed-content requests
>   to `http://` are blocked by browsers.

### 3. Reload Apache
```shell
# Validate the config first, then reload (graceful, no dropped connections)
apachectl configtest        # -> "Syntax OK"
systemctl reload httpd
# or: apachectl -k graceful
```

### 4. Verify from the client
```shell
# (a) Range request must return 206 Partial Content + Content-Range
curl -I -H "Range: bytes=0-999" \
  https://neomorph.salk.edu/ftp/bican/hg38_with_chrL.allc.cz
# expected: HTTP/1.1 206 Partial Content
#           Accept-Ranges: bytes
#           Content-Range: bytes 0-999/<total>

# (b) CORS pre-flight must echo the Access-Control-* headers
curl -I -X OPTIONS \
  -H "Origin: https://dingwb.github.io" \
  -H "Access-Control-Request-Method: GET" \
  https://neomorph.salk.edu/ftp/bican/hg38_with_chrL.allc.cz
# expected: Access-Control-Allow-Origin: *
#           Access-Control-Expose-Headers: Content-Range, Accept-Ranges, Content-Length
```

Once both checks pass, `czip query -I https://neomorph.salk.edu/...`, the
Python `Reader.from_url()`, and the browser `cz_viewer.html` can all read the
files directly without a proxy.

## docs
```shell
pip install sphinx sphinx-autobuild sphinx-rtd-theme sphinx-copybutton pandoc nbsphinx sphinx_pdj_theme sphinx_sizzle_theme recommonmark readthedocs-sphinx-search
conda install conda-forge::pandoc

mkdir -p docs && cd docs
sphinx-quickstart
# Separate source and build directories (y/n) [n]: y
# Project name: adataviz

# vim source/conf.py
# add *.rst

# cd docs
# vim index.html: <meta http-equiv="refresh" content="0; url=./build/html/index.html" />
# make sure the env that has sphinx 7.x + recommonmark is active (NOT base conda)
# conda activate /home/x-wding2/Software/conda/m3c
# setting - pages: deploy from branch -> Github Actions
cd docs
rm -rf build
ln -sfn ~/Projects/Github/cytozip/notebooks source/notebooks
sphinx-apidoc -e -o source -f ../../cytozip ../../cytozip/setup.py
CC=gcc CXX=g++ make html
rm -rf source/notebooks
cd ..
ls
ls docs

# after modifying cz_viewer.html
cp docs/source/_static/cz_viewer.html docs/build/html/_static/

vim .nojekyll #create empty file
```

## Run test & notebooks
```shell
cd /home/x-wding2/Projects/Github/cytozip
pip uninstall -y cytozip && python3 -m pip install .
rm -rf cytozip_example_data/output/cz
czip build_ref -g ~/Ref/mm10/mm10_ucsc_with_chrL.fa -O cytozip_example_data/output/mm10_with_chrL.allc.cz -j 20
python tests/benchmark_bam_to_cz.py  -j 1
python tests/benchmark_query.py
# nbexe notebooks/2.dnam.ipynb
# upload to figshare
figshare upload -i cz/ --title cytozip_example_data -d "cytzip example datasets" --target_folder cz -W 4 --overwrite
```

## Cytozip Genome Browser

The browser (`docs/source/_static/cz_viewer.html`) pulls two kinds of data from the
**WashU Epigenome Browser** infrastructure:

- **Reference DNA sequence** as UCSC `.2bit` files hosted at
  `https://vizhub.wustl.edu/public/<genome>/<genome>.2bit` (read in-browser with
  the JBrowse [`@gmod/twobit`](https://www.npmjs.com/package/@gmod/twobit) reader).
- **Gene annotations** (refGene / GENCODE) via the WashU REST API
  `https://lambda.epigenomegateway.org/v3/<genome>/genes/<track>/queryRegion`.

### How to cite

If you use these genome-sequence and gene-annotation resources, please cite the
**WashU Epigenome Browser** (cite the most recent update paper; the original 2011
paper is optional as the founding reference):

- Seng C, Liu S, Zhang W, Zhuo X, Li D, Wang T. **WashU Epigenome Browser update 2025.**
  *Nucleic Acids Research.* 2025 Jul 7;53(W1):W554–W561. doi:[10.1093/nar/gkaf387](https://doi.org/10.1093/nar/gkaf387). PMID: 40322916.
- Li D, Purushotham D, Harrison JK, Hsu S, Zhuo X, Fan C, Liu S, Xu V, Chen S, Xu J,
  Ouyang S, Wu AS, Wang T. **WashU Epigenome Browser update 2022.**
  *Nucleic Acids Research.* 2022 Jul 5;50(W1):W774–W781. doi:[10.1093/nar/gkac238](https://doi.org/10.1093/nar/gkac238). PMID: 35412637.
- Li D, Hsu S, Purushotham D, Sears RL, Wang T. **WashU Epigenome Browser update 2019.**
  *Nucleic Acids Research.* 2019 Jul 2;47(W1):W158–W165. doi:[10.1093/nar/gkz348](https://doi.org/10.1093/nar/gkz348). PMID: 31165883.
- Zhou X, Maricque B, Xie M, Zhang D, Sundaram V, Edwards CA, Wang T.
  **The Human Epigenome Browser at Washington University.**
  *Nature Methods.* 2011 Nov 22;8(12):989–990. doi:[10.1038/nmeth.1772](https://doi.org/10.1038/nmeth.1772). PMID: 22127213.

### Underlying data sources (cite if relevant to your analysis)

- **Gene models** — RefSeq (`refGene`) and/or GENCODE, depending on the annotation
  set you display:
  - RefSeq: O'Leary NA, *et al.* **Reference sequence (RefSeq) database at NCBI.**
    *Nucleic Acids Research.* 2016;44(D1):D733–D745. doi:[10.1093/nar/gkv1189](https://doi.org/10.1093/nar/gkv1189).
  - GENCODE: Frankish A, *et al.* **GENCODE: reference annotation for the human and
    mouse genomes in 2023.** *Nucleic Acids Research.* 2023;51(D1):D942–D949. doi:[10.1093/nar/gkac1071](https://doi.org/10.1093/nar/gkac1071).
- **Genome assemblies** — GRCh38/hg38, GRCh37/hg19 (human) and GRCm39/mm39,
  GRCm38/mm10 (mouse) from the Genome Reference Consortium (distributed via UCSC as
  `.2bit`).

## upload to conda
```shell
# 1. Fork https://github.com/bioconda/bioconda-recipes
git clone https://github.com/DingWB/bioconda-recipes.git
cd bioconda-recipes
git checkout -b add-cytozip # create a new branch


# 2. add recipe
mkdir -p recipes/cytozip
cp /home/x-wding2/Projects/Github/cytozip/conda-recipe/meta.yaml recipes/cytozip/

# Calculate real sha256（release the first version to PyPI）：
curl -sL https://pypi.io/packages/source/c/cytozip/cytozip-0.3.7.tar.gz | sha256sum
# fill in sha256 to recipes/cytozip/meta.yaml

# test before PR
# mamba create -n bioconda-build -c conda-forge -c bioconda conda-build boa conda-verify
# conda mambabuild -c conda-forge -c bioconda recipes/cytozip
mamba create -n bioconda-build -c conda-forge -c bioconda bioconda-utils conda-build conda-forge-pinning anaconda-client
conda activate bioconda-build
bioconda-utils lint --packages cytozip
bioconda-utils build --packages cytozip
# If you see .conda / .tar.bz2 in output folder, then recipe is good, then:
git add recipes/cytozip/meta.yaml
git commit -m "Add cytozip 0.3.5"
git push origin add-cytozip
# PR to bioconda/bioconda-recipes:master, and finally:
conda install -c bioconda cytozip
```

### Upload to your own conda channel
```shell
conda create -y -p ~/Software/conda/czbuild -c conda-forge \
    conda-build conda-forge-pinning anaconda-client
# update meta.yaml (version & sha256sum)
curl -sL https://pypi.io/packages/source/c/cytozip/cytozip-0.3.9.tar.gz | sha256sum

conda activate ~/Software/conda/czbuild
rm -rf conda-build
for PY in 3.10 3.11 3.12 3.13; do
  ~/Software/conda/czbuild/bin/conda build conda-recipe --python $PY -c conda-forge -c bioconda --output-folder conda-build
done
# anaconda login # or use token: export ANACONDA_API_TOKEN=
anaconda upload conda-build/*/cytozip-*.conda -u wubinding

conda install -c wubinding -c bioconda cytozip
```

## Cell-type prediction (`cytozip/model.py`)

`model.py` provides a **site-likelihood (naive-Bayes) cell-type classifier**
for shallow snm3C-seq. Given a query cell's per-cytosine `mc`/`cov` `.cz` and a
set of cell-type **pseudobulk** `.cz` (deep reference), it predicts the most
likely cell type.

### Method in one paragraph

For every reference cytosine `c` and cell type `t`, the pseudobulk methylation
frequency is a Beta-shrinkage estimate
`theta[c,t] = (mc + alpha0) / (cov + alpha0 + beta0)` (kept continuous, never
binarized). A query cell with `mc`/`cov` at each site scores each type by the
aggregated per-site Bernoulli log-likelihood
`sum_c ( mc*log(theta) + (cov-mc)*log(1-theta) )`. CpG and CpH are modelled as
two independent **channels** (own frequencies + shrinkage prior), summed in
log-space with weights `lambda_cg`/`lambda_ch`, plus a class log-prior; a
softmax over types gives calibrated probabilities. The CpG/CpH split comes from
the `context` column of a `build_ref` reference `.cz`.

Only **discriminative** sites are kept: the per-site score is the across-type
range `theta.max - theta.min`; sites with score `== 0` (all types share the
same theta) are dropped because they add an identical constant to every type
and cancel in the softmax — dropping them is lossless. Use `top_cg`/`top_ch`
(keep top-N) or `min_range_cg`/`min_range_ch` (keep score > threshold) to prune
harder and shrink the model.

### Two entry points

- **`CellTypeClassifier`** — the class: `fit()` → `save()` / `load()` →
  `predict()` / `predict_batch()` / `predict_multicell()` /
  `predict_proba()` / `log_posterior()` / `site_importance()`.
- **`predict_cell_type(...)`** — a one-shot wrapper that fits (or reuses a
  saved model), predicts, and (with `outdir=`) writes everything to disk.

### Quick start

```python
from cytozip.model import CellTypeClassifier, predict_cell_type

# --- A) one-shot: fit + predict + write outputs ---
labels, proba = predict_cell_type(
    query="cells_dir_or_table_or.cz",     # see "query forms" below
    pseudobulks="pseudobulk_dir/",        # or {cell_type: path} dict
    reference="ref.cz",                   # build_ref .cz (pos/strand/context)
    top_cg=50000, top_ch=50000,           # prune to keep the model small
    n_jobs=-1,
    outdir="~/out")                       # writes model + predictions here

# --- B) explicit: fit once, save, reuse ---
clf = CellTypeClassifier(lambda_cg=1.0, lambda_ch=1.0)
clf.fit(pseudobulks="pseudobulk_dir/", reference="ref.cz",
        top_cg=50000, top_ch=50000, outdir="~/out/model")
clf.save("~/out/model")

clf = CellTypeClassifier.load("~/out/model")   # arrays are mmap-ed, low RAM
res = clf.predict("one_cell.cz")               # -> dict(label, confidence, proba, log_posterior)
labels, proba = clf.predict_batch(["a.cz", "b.cz"], n_jobs=-1)
```

**`query` forms** (auto-detected): a single `.cz` (→ one prediction `dict`); a
**concatenated multi-cell** `.cz` (a `catcz` output; → `(labels, proba)`); a
**directory** of `.cz`; a 2-column `[cell_id, cz_path]` **table** (file or
DataFrame); a `{cell_id: path}` dict or list of paths; or a preloaded
`(mc, cov)` array pair. Local paths and **remote URLs** (`http(s)://`,
`s3://`, `gs://`, `hf://`, ...) both work.

### What `outdir/` contains

| Path | What it is |
|------|-----------|
| `<outdir>/model/` | the saved model store (see below). If it already exists and is complete, `predict_cell_type` **skips fitting** and just loads it — so `pseudobulks`/`reference` need not be re-supplied. |
| `<outdir>/predictions.csv` | one row per cell, index `cell_id`, columns `label`, `confidence` (the top softmax probability; `label='unassigned'` if below `abstain_threshold`). |
| `<outdir>/predict_proba.csv` | cell × cell_type matrix of softmax posterior probabilities (rows sum to 1), index `cell_id`. |

> The model store is written **directly** under `<outdir>/model` (real disk).
> Without `outdir` it falls back to `$TMPDIR`/`/tmp`, which on HPC is often
> RAM-backed (tmpfs) and can crash a large model with a **"Bus error"
> (SIGBUS)** — pass `outdir=` on scratch/large disk if you hit that.

### `model/` directory — file formats

A saved model is a small **self-contained directory** of uncompressed NumPy
arrays plus one JSON. Arrays are stored uncompressed on purpose so `load()` can
**memory-map** them (`mmap_mode='r'`) and score chunk-by-chunk without ever
holding the full `(n_sites × n_types)` table in RAM.

```
model/
├── meta.json            # all scalars + axis + per-channel layout
├── cg_log_theta.npy     # float32 (n_cg,  n_types)  = log(theta)   CpG channel
├── cg_log1m_theta.npy   # float32 (n_cg,  n_types)  = log(1-theta) CpG channel
├── cg_sites.npy         # int32   (n_cg,)           reference-row index per site
├── ch_log_theta.npy     # float32 (n_ch,  n_types)  = log(theta)   CpH channel
├── ch_log1m_theta.npy   # float32 (n_ch,  n_types)  = log(1-theta) CpH channel
└── ch_sites.npy         # int32   (n_ch,)           reference-row index per site
```

Only the channels that have sites are written (a CG-only model has no `ch_*`).

**`meta.json` fields**

| Field | Meaning |
|-------|---------|
| `format` / `version` | `"cytozip.CellTypeClassifier"` / schema version (`3`). |
| `alpha0_cg`,`beta0_cg`,`alpha0_ch`,`beta0_ch` | the Beta shrinkage prior actually used per channel. |
| `lambda_cg`,`lambda_ch` | CpG/CpH log-space channel weights. |
| `mc_col`,`cov_col`,`context_col` | column names expected in the `.cz` inputs. |
| `cell_types` | ordered list of class names → this is the **column order** of the `*_log_theta.npy` matrices and of `predict_proba`. |
| `cell_counts` | per-type reference cell counts (or `null`) for the abundance prior. |
| `chunk_keys` | ordered list of chunk keys, e.g. `[["chr1"], ["chr2"], ...]` — the reference axis order. |
| `chunk_lens` | per-chunk reference row count (aligned to `chunk_keys`); `sum == n_full`. |
| `n_full` | total number of reference cytosines (the full all-C axis length). |
| `cg`,`ch` | per-channel layout (or `null`): `alpha0`,`beta0`,`n_sites`, and `chunks = {chunk_id: [lo, hi]}` where `chunk_id` is the **index into `chunk_keys`** and `[lo, hi)` is the row range of that chunk inside the channel's concatenated arrays. |

**How to read the arrays directly**

```python
import json, numpy as np, os
d = os.path.expanduser("~/out/model")
meta = json.load(open(f"{d}/meta.json"))
types = meta["cell_types"]                    # column order

lt = np.load(f"{d}/cg_log_theta.npy", mmap_mode="r")   # (n_cg, n_types) float32
si = np.load(f"{d}/cg_sites.npy")                       # (n_cg,) int32

theta = np.exp(lt[:])                          # methylation frequency per site×type
# site i belongs to chromosome/chunk via meta['cg']['chunks']:
#   for chunk_id, (lo, hi) in meta['cg']['chunks'].items():
#       chrom = meta['chunk_keys'][int(chunk_id)][0]
#       si[lo:hi] are that chrom's reference-row indices for these sites
```

- `*_log_theta.npy[i, t]` = `log(theta[site i, type t])` → `exp` to get the
  methylation frequency (0–1) of cell type `t` at that site.
- `*_log1m_theta.npy` = `log(1 - theta)` (precomputed only to speed scoring;
  it is recoverable as `log1p(-exp(log_theta))`).
- `*_sites.npy[i]` = the **row index within that chromosome's reference axis**
  of site `i` (i.e. which reference cytosine). Map to a genomic coordinate by
  reading the same `build_ref` reference `.cz` chunk at that row (or, more
  conveniently, use `site_importance(reference=...)`).

Because `load()` only mmaps these, a model can be **10s of GB on disk yet score
with a small RAM footprint**; pruning at fit time (`top_*` / `min_range_*`) is
the main lever to shrink both.

### Biological interpretation — `site_importance()`

`clf.site_importance(reference=None, top=None, include_theta=True)` ranks the
model's kept cytosines by how discriminative they are and returns a
`pandas.DataFrame` sorted by importance. **Importance is exactly the signal
`fit()` selects on**: the across-cell-type methylation-frequency range
`theta.max - theta.min`. Large values = differentially methylated across cell
types (the biologically meaningful markers).

```python
imp = clf.site_importance(reference="ref.cz", top=1000)   # top 1000 markers
imp.to_csv("top_sites.csv", index=False)
```

Output columns:

| Column | Meaning |
|--------|---------|
| `context` | `CG` or `CH`. |
| `chrom` | chromosome (from the model's chunk key). |
| `ref_row` | index within that chromosome's reference axis (= `*_sites.npy` value). |
| `importance` | `theta.max - theta.min` across cell types (bigger = more discriminative). |
| `top_type` | cell type with the **highest** methylation at that site. |
| `min_type` | cell type with the **lowest** methylation there. |
| `pos`,`strand`,`ref_context` | genomic coordinate / context (only when `reference=` is given). |
| one column per cell type | that type's `theta` (methylation frequency), when `include_theta=True`. |

Typical uses: group by `context` for CG vs CH markers; annotate `pos` against
genes/enhancers/DMRs; for a target cell type, filter `top_type == <type>` with
high `importance` to get that type's characteristic (hyper-methylated) markers.
For large models pass `top=` to bound the table.

### Per-cell-type motif FASTA — `top_cytosine_fasta()`

`top_cytosine_fasta(importance, genome, ...)` turns a `site_importance` table
(built with `reference=` so it has `pos`/`strand`) into flanking-sequence
FASTA: for each cell type and each context (CG / CH), it takes the `top_n` most
important cytosines and extracts `flank` bp on each side from a genome FASTA
(minus-strand sites are reverse-complemented so the C stays centred). Feed the
per-type files to a motif finder (MEME/HOMER).

```python
from cytozip.model import top_cytosine_fasta

imp = clf.site_importance(reference="ref.cz")            # must include pos/strand
top_cytosine_fasta(
    imp, genome="genome.fa",          # indexed (.fai beside it) or pysam.FastaFile
    flank=50, top_n=200,              # window = 2*flank+1 bp, top 200 per (type, context)
    group_col="top_type",             # or "min_type"
    out_fasta="~/out/top_motifs.fa",  # one combined FASTA
    split_dir="~/out/motifs")         # + one FASTA per <cell_type>.<context>.fa
```

FASTA headers encode everything: `>{cell_type}|{context}|{chrom}:{pos}:{strand}|imp={importance}`.
Needs `pysam` (reads the genome via a `.fai` index); `pos_base` defaults to 1
(1-based coordinates), and edge windows are dropped unless
`drop_incomplete=False`.

### Other predict-time options

- `max_query_cg` / `max_query_ch` (on `predict*`): randomly keep at most that
  many of the query's **covered** cytosines per channel (downsampling); `None`
  uses all. Re-drawn per cell.
- `prior_alpha` + `cell_counts`: temper the abundance prior (`0` = uniform).
- `abstain_threshold`: label a cell `'unassigned'` when top probability is below it.

## Bulk deconvolution (`deconvolve` / `deconvolve_bulk`)

The **same** cell-type pseudobulk reference used for classification can also
**deconvolve a bulk sample into cell-type fractions**. Instead of asking "which
single type is this cell?", deconvolution asks "what mixture of cell types
produced this bulk?" — given a bulk **WGBS** `.cz` (or a methylation-**array**
beta profile) and the per-type reference frequencies `theta[c,t]`, it solves
for fractions `f_t` such that `bulk_beta[c] ≈ sum_t f_t * theta[c,t]`.

### Method in one paragraph

At the model's discriminative (marker) sites, the bulk methylation level
`beta[c] = mc/cov` is modelled as a **coverage-weighted linear mixture** of the
reference cell types. The fractions are found by **constrained least squares**:
non-negativity (`f_t ≥ 0`) via NNLS, plus (by default) a **sum-to-one**
equality `sum_t f_t = 1` via SLSQP — a Houseman/CIBERSORT-style reference-based
deconvolution. Deep sites weigh more (`weight = cov`); a methylation array
(beta only, no coverage) falls back to ordinary least squares.

### Entry points

- **`CellTypeClassifier.deconvolve()`** — one bulk → a `pandas.Series` of
  fractions (index = cell types); `.attrs['r2']` / `.attrs['n_sites']` report
  fit quality. Also `deconvolve_batch()` (many bulks → a fractions DataFrame)
  and `deconvolve_multicell()` (many bulks packed in one cat `.cz`).
- **`deconvolve_bulk(...)`** — one-shot wrapper that fits (or reuses a saved
  model) and deconvolves, mirroring `predict_cell_type(...)`.

### Quick start

```python
from cytozip.model import CellTypeClassifier, deconvolve_bulk

# --- A) one-shot: fit reference + deconvolve + write fractions.csv ---
frac = deconvolve_bulk(
    query="bulk_wgbs.cz",                 # a .cz / dir / table / cat .cz / beta array
    pseudobulks="pseudobulk_dir/",        # or {cell_type: path} dict
    reference="ref.cz",                   # build_ref .cz (pos/strand/context)
    contexts="cg",                        # "cg" (default) | "ch" | "cg+ch"
    top_cg=50000,                         # prune markers to keep the model small
    outdir="~/out")                       # writes model + fractions.csv here

# --- B) explicit: reuse a fitted/saved model ---
clf = CellTypeClassifier.load("~/out/model")     # mmap-ed, low RAM
frac = clf.deconvolve("bulk_wgbs.cz", contexts="cg")
print(frac)                    # Series: fraction per cell type (sums to 1)
print(frac.attrs["r2"], frac.attrs["n_sites"])   # fit quality

# many bulks at once -> DataFrame (rows = samples, cols = types + r2 + n_sites)
frac_df = clf.deconvolve_batch(["bulkA.cz", "bulkB.cz"], contexts="cg", n_jobs=-1)
```

### Options

| Option | Meaning |
|--------|---------|
| `contexts` | `'cg'` (default; the only choice for arrays, and the usual one for WGBS), `'ch'`, or `'cg+ch'` to use both channels jointly. |
| `weight_by_cov` | weight each site by its bulk coverage (weighted LS). Default `True`; ignored for array beta input (no coverage). |
| `sum_to_one` | constrain fractions to sum to 1 (default `True`). `False` → plain NNLS (fractions need not sum to 1). |
| `allow_unknown` | with `sum_to_one=True`, relax to `sum ≤ 1` and report the remainder as an extra **`'unknown'`** fraction (an unmodelled compartment). |
| `min_cov` | only use bulk sites with `cov ≥ this` (default 1). |
| `n_jobs` | threads for `deconvolve_batch` / `deconvolve_multicell`. |

### Methylation-array (beta) input

An array (e.g. Illumina 450K/EPIC) has no coverage — pass a **full-axis 1-D
beta `np.ndarray`** aligned to the reference axis; it is treated as unit
coverage (`beta = mc/cov`), so weighting falls back to ordinary least squares.
Use `contexts='cg'` (arrays are CpG-only).

```python
import numpy as np
beta = np.load("array_beta_full_axis.npy")   # length == n reference cytosines
frac = clf.deconvolve(beta, contexts="cg")
```

### `query` forms & outputs

`deconvolve` / `deconvolve_bulk` accept the same auto-detected `query` forms as
`predict_cell_type` — a single `.cz`, a **cat** multi-sample `.cz`, a
**directory**, a `[sample_id, cz_path]` **table**, a `{sample_id: path}` dict /
list, a preloaded `(mc, cov)` pair, or (bulk-only) a full-axis **beta** array.
A single sample returns a `Series`; anything batch-like returns a DataFrame
(rows = samples; columns = cell-type fractions [+ `'unknown'`] then `r2`,
`n_sites`). With `outdir=`, fractions are written to `<outdir>/fractions.csv`
and the fitted model reused from `<outdir>/model` if already complete.

## Peak calling from methylation (`call_peaks` / `call_peaks_bdg`)

CpG **hypomethylation** marks open chromatin / regulatory elements, so the
per-site **unmethylated count** `umc = cov - mc` can be used as an ATAC-like
signal and fed to **MACS3** for peak calling. Two routes are provided (both in
`cytozip/dmr.py`), and both run on a **pseudobulk** single-track `.cz` (merge
same-type cells first with `merge_cz`; single cells are too sparse).

> **Always pass a coverage control** (`control='cov'`). `umc` is confounded by
> depth and CpG density; using total `cov` as the MACS3 input track makes peaks
> reflect *local unmethylation enrichment* (`umc/cov` above the global rate),
> not just deeply covered / CpG-island regions.

### Two routes

| | `call_peaks` (pseudo-reads) | `call_peaks_bdg` (bedGraph) |
|---|---|---|
| How | expands each site into `umc` pseudo-reads → `macs3 callpeak` | difference-array pileup → `macs3 bdgopt`→`bdgcmp`→`bdgpeakcall` |
| Memory | grows with `sum(umc)` (large on deep pseudobulks) | `O(n_sites)`, independent of `sum(umc)` |
| Output | `<name>_peaks.narrowPeak` (+ MACS3 files) | `<name>_peaks.narrowPeak` |
| Use when | small/medium tracks, want full MACS3 output | deep pseudobulks (recommended) |

### Quick start

```python
import cytozip as czip

# A) pseudo-read route + coverage control
czip.call_peaks(
    input="pseudobulk.cz", reference="mm10.allc.cz",
    index="mm10.CGN.index",        # CpG-only (CpH is a different signal)
    control="cov",                 # coverage control track (macs3 -c) — recommended
    genome_size="mm", fragment_size=300, qvalue=0.05,
    name="celltypeA")

# B) memory-efficient bedGraph route (preferred for deep pseudobulks)
czip.call_peaks_bdg(
    input="pseudobulk.cz", reference="mm10.allc.cz",
    index="mm10.CGN.index", control="cov",
    ext=300, method="ppois", cutoff=2.0,   # cutoff = -log10(p) for ppois
    name="celltypeA")
```

CLI:

```bash
czip call_peaks     -I pseudobulk.cz -r mm10.allc.cz -s mm10.CGN.index \
                    --control cov -g mm -n celltypeA
czip call_peaks_bdg -I pseudobulk.cz -r mm10.allc.cz --index mm10.CGN.index \
                    --control cov --ext 300 --cutoff 2.0 -n celltypeA
```

### Key options

| Option | Meaning |
|--------|---------|
| `signal` | `'unmeth'` = `cov-mc` (default, open-chromatin proxy) or `'meth'` = `mc`. |
| `control` | coverage-bias control track: `'cov'` (recommended) / `'mc'`. `call_peaks` also accepts `None` (genome background only, less specific); `call_peaks_bdg` always uses one. |
| `index` | context-filter index (e.g. a CpG-only `index_context` output) — use CpG for open chromatin. |
| `min_cov` | drop sites with `cov <` this (default 1). |
| `fragment_size` / `ext` | bp each site's count is spread over (default 300). |
| `qvalue` (`call_peaks`) | MACS3 `callpeak -q` cutoff. |
| `method` / `cutoff` / `min_len` / `max_gap` (`call_peaks_bdg`) | `bdgcmp` score method (`ppois`…), `bdgpeakcall` score cutoff, min peak length (default `ext`), max merge gap (default `ext//2`). |

`to_bedgraph(...)` additionally dumps a per-site `unmeth` / `meth` /
`frac_unmeth` bedGraph for browsers or manual `macs3 bdgpeakcall`.

> MACS3's Poisson model is an approximation to `umc ~ Binomial(cov, 1-p)`, so
> treat the reported p/q values as ranking thresholds, not exact FDR.

## To-Do List
- [x] Peak calling using umc (see "Peak calling from methylation" above)
- [x] Cell type prediction using Bayes model (see "Cell-type prediction" above)
- [x] Bulk deconvolution into cell-type fractions (see "Bulk deconvolution" above)
- Online cz visualizing tool: modify pyGenomeTrack to support .cz file
