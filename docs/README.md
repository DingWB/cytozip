# for developer

```shell
cd docs
rm -rf build
ln -s ~/Projects/Github/czip/notebooks/ source/notebooks
sphinx-apidoc -e -o source -f ../../cytozip
make html
rm -rf source/notebooks
cd ..
ls
```

```shell
rm -rf dist && rm -rf cytozip.egg-info/
python setup.py sdist bdist_wheel
twine upload dist/*
```

### merge allc using cytozip

```shell
# cytozip merge allc
time cytozip merge_mz -i mz -o merged_all.cz -n 20 -P ~/Ref/mm10/mm10_ucsc_with_chrL.main.chrom.sizes.txt
# cytozip mergemz (only merge CG 1m54.782s, 20 cpu)
time cytozip merge_mz -i mz-CGN -o merged.cz -n 20 -k False
# view 
cytozip Reader -I merged.cz view -s 0 -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz |head

# Local: merge cell types (Major Type)


# Local snakemake
snakemake --printshellcmds -s ~/MySnakemake/workflow/rules/merge_mz.smk --config indir=/anvil/scratch/x-wding2/Projects/mouse_pfc/scripts/pseudo_cell/mz-CGN cell_table=test_cell_table.tsv chrom=/home/x-wding2/Ref/mm10/mm10_ucsc_with_chrL.main.chrom.sizes.txt cores=96 -j 1
# GCP snakemake
snakemake --printshellcmds -s ~/MySnakemake/workflow/rules/merge_mz.smk --config indir=pfc_mz-CGN cell_table=cell_table.tsv chrom=mm10_ucsc_with_chrL.main.chrom.sizes.txt chrom_prefix=gs://wubin_ref/mm10 cores=64 gcp=True --default-remote-prefix mouse_pfc --default-remote-provider GS --google-lifesciences-region us-west1 --scheduler greedy -j 32
```

### merge cell types (MajorType)

sum up the mc and cov

```shell
mkdir -p MajorType
cd MajorType
sed '1d' ~/Projects/mouse_pfc/5kb_mC_hic_clustering/L2/MajorTypes.tsv > cell_table.tsv
cytozip merge_cell_type -i ~/Projects/mouse_pfc/cz-CG -c cell_table.tsv -o ~/Projects/mouse_pfc/pseudo_cell/MajorType -n 96 -P ~/Ref/mm10/mm10_ucsc.main.chrom.sizes -e ".cz"
```

### merge cell types (Cell Class)

```shell
mkdir -p CellClass
cd CellClass

sed '1d' ~/Projects/mouse_pfc/5kb_mC_hic_clustering/L2/major_type_to_cell_class.tsv > cell_table.tsv
cytozip merge_cell_type -i ~/Projects/mouse_pfc/pseudo_cell/MajorType/cz -c cell_table.tsv -o ~/Projects/mouse_pfc/pseudo_cell/CellClass/cz -n 96 -P ~/Ref/mm10/mm10_ucsc.main.chrom.sizes -e ".cz"
```

### merge multiple .cz files into matrix (fraction, 2D or fisher)

```shell
mkdir matrix
cytozip merge_cz -i cz -n 96 -f 2D -P ~/Ref/mm10/mm10_ucsc_with_chrL.main.chrom.sizes.txt -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz
cytozip merge_cz -i cz -n 96 -f fraction -o matrix/major_type.beta.bed -P ~/Ref/mm10/mm10_ucsc.main.chrom.sizes  -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz

cytozip merge_cz -i cz -n 96 -f fisher -o matrix/major_type.fisher.bed -P ~/Ref/mm10/mm10_ucsc.main.chrom.sizes -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz
```

### comb-p call DMR (MajorType)

```shell
cytozip combp -i matrix/major_type.fisher.bed.gz -o DMR_cpv -n 96
```

### convert  MajorType .cz to allc.tsv.gz

```shell
indir="/anvil/scratch/x-wding2/Projects/mouse_pfc/pseudo_cell/MajorType/cz"
outdir="/anvil/scratch/x-wding2/Projects/mouse_pfc/pseudo_cell/MajorType/allc"
mkdir -p ${outdir}
for file in `find ${indir} -name "*.cz"`; do
  bname=$(basename $file)
  sname=${bname/.cz/}
  echo ${sname}
#  mv ${sname}.cz ${sname}.cz
  echo "cytozip Reader -I ${indir}/${sname}.cz view -s 0 -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz -h False | awk 'BEGIN{FS=OFS=\"\t\"}; {print \$0,1}' | bgzip > ${outdir}/${sname}.allc.tsv.gz && tabix -f -s 1 -b 2 -e 2 ${outdir}/${sname}.allc.tsv.gz" >> cmd
done;
cat cmd |parallel -j 30
```

### merge Cell Class multiple .cz files into matrix (fraction)

```shell
mkdir -p matrix
cytozip merge_cz -i cz -n 96 -f fraction -o matrix/cell_class.beta.bed -P ~/Ref/mm10/mm10_ucsc.main.chrom.sizes -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz

# run fisher between cell class
cytozip merge_cz -i cz -n 96 -f fisher -o matrix/cell_class.fisher.bed -P ~/Ref/mm10/mm10_ucsc.main.chrom.sizes -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz

# run fisher for each cell_class (within class), not between class
cytozip merge_cz -i ~/Projects/mouse_pfc/pseudo_cell/MajorType/cz -n 128 -f fisher -p matrix/cell_class.fisher -P ~/Ref/mm10/mm10_ucsc.main.chrom.sizes -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz --class_table cell_table.tsv

cytozip merge_cz -i ~/Projects/mouse_pfc/pseudo_cell/MajorType/cz -n 128 -f fraction -p matrix/cell_class.beta -P ~/Ref/mm10/mm10_ucsc.main.chrom.sizes -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz --class_table cell_table.tsv
```

### comb-p call DMR

```shell
# between CellClass
cytozip combp -i matrix/cell_class.fisher.bed.gz -o between_groups_dmr/cpv -n 72
cytozip combp -i matrix/cell_class.fisher.Exc.txt.gz -o within_groups_dmr/Exc -n 72
cytozip combp -i matrix/cell_class.fisher.Inh.txt.gz -o within_groups_dmr/Inh -n 72
cytozip combp -i matrix/cell_class.fisher.NonN.txt.gz -o within_groups_dmr/NonN -n 72
cytozip combp -i matrix/cell_class.fisher.RG.txt.gz -o within_groups_dmr/RG -n 72

# agg dmr beta
cytozip intersect -Q merged_dmr.txt -M ../matrix/cell_class.beta.bed.gz -O merged_dmr.cell_class.beta.txt -m False
cytozip intersect -Q merged_dmr.txt -M /anvil/scratch/x-wding2/Projects/mouse_pfc/pseudo_cell/MajorType/matrix/major_type.beta.bed.gz -O merged_dmr.major_type.beta.txt -m False
# annot cpv dmr
cytozip annot_dmr -i merged_dmr.txt -m merged_dmr.cell_class.beta.txt -o dmr.annotated.txt
```

### convert  Cell Class .cz to allc.tsv.gz

```shell
indir="/home/x-wding2/Projects/mouse_pfc/pseudo_cell/CellClass/cz"
cd ${indir}
outdir="/home/x-wding2/Projects/mouse_pfc/pseudo_cell/CellClass/allc"
mkdir -p ${outdir}
for file in `find ${indir} -name "*.cz"`; do
  bname=$(basename $file)
  sname=${bname/.cz/}
  echo ${sname}
  echo "cytozip Reader -I ${indir}/${sname}.cz view -s 0 -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz -h False | awk 'BEGIN{FS=OFS=\"\t\"}; {print \$0,1}' | bgzip > ${outdir}/${sname}.allc.tsv.gz && tabix -f -s 1 -b 2 -e 2 ${outdir}/${sname}.allc.tsv.gz" >> cmd
done;
cat cmd |parallel -j 64
```