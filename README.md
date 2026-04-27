# Cytozip: Chunk based ZIP for cytosine data

## Installation

### Recommended: conda (brings the C libraries with it)

cytozip ships two Cython extensions that link against external C
libraries:

- `cytozip.cz_accel` — depends on **libdeflate**
- `cytozip._bam_pileup` *(optional, but strongly recommended)* —
  depends on **htslib**, provides a fast in-process BAM → cz pileup
  backend (~10-20× faster than shelling out to `samtools mpileup`,
  byte-equivalent output).

`pip` cannot install these C libraries, so the easiest path is conda:

```shell
# (when published to bioconda)
conda install -c bioconda -c conda-forge cytozip

# from source via the included recipe:
conda build conda-recipe -c bioconda -c conda-forge
conda install -c local cytozip
```

### pip

```shell
# Prerequisites (one of):
#   conda install -c bioconda htslib libdeflate          # recommended
#   apt-get install libhts-dev libdeflate-dev            # Debian/Ubuntu
#   brew install htslib libdeflate                       # macOS
pip install cytozip
# or
pip uninstall -y cytozip & pip install git+http://github.com/DingWB/cytozip
```

If `htslib` headers are not found at build time, the `_bam_pileup`
extension is **silently skipped** and `bam_to_cz()` automatically falls
back to the `samtools mpileup` subprocess backend (slower, but
identical output). `libdeflate` is required (not optional).

reinstall
```shell
pip uninstall -y cytozip && pip install git+http://github.com/DingWB/cytozip
```

### BAM pileup backend selection

`cytozip.bam.bam_to_cz` picks a backend at runtime in this order:

1. **htslib** (in-process, default if `_bam_pileup` was built)
2. **`samtools mpileup`** subprocess — forced via
   `CYTOZIP_BAM_BACKEND_MPILEUP=1`, or used automatically when htslib
   is unavailable.


## Implementation
|                                  | allcools | ballcools | cytozip |
| -------------------------------- | -------- | --------- | ----- |
| Format                           | .tsv.gz  | .ballc    | .cz   |
| Compression algorithm            | bgzip    | bgzip     | cytozip |
| Support Random Access ?          | Yes      | Yes       | Yes   |
| Need extra index file for query? | Yes      | yes       | No    |
| Quickly Merge?                   | No       | No        | Yes   |

![img.png](docs/images/tab1.png)
<!---
![docs/images/img.png](docs/images/design.png)
-->
## Usage

[Documentation](https://dingwb.github.io/cytozip)

## Example dataset

[https://figshare.com/articles/dataset/cytozip_example_data/25374073](https://figshare.com/articles/dataset/cytozip_example_data/25374073)


## dev
```shell
python setup.py build_ext --inplace
```