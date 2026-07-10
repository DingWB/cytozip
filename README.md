# cytozip: index-free, chunk-compressed storage for scalable single-cell DNA methylation analysis
A scalable, cloud-native ecosystem for single-cell and population DNA methylation analysis
## Installation
### pip

```shell
# Prerequisites (one of):
#   conda install -c bioconda htslib libdeflate          # recommended
#   apt-get install libhts-dev libdeflate-dev            # Debian/Ubuntu
#   brew install htslib libdeflate                       # macOS
pip install cytozip
# or reinstall
pip uninstall -y cytozip && pip install git+http://github.com/DingWB/cytozip
```

### conda (To-DO)
```shell
# (when published to bioconda)
conda install -c bioconda -c conda-forge cytozip
```

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
