import os.path
import pandas as pd
import os

if 'outdir' in config:
    outdir = config["outdir"]  #
else:
    outdir = 'allc-CG'

if 'gcp' in config and config["gcp"]:
    from snakemake.remote.GS import RemoteProvider as GSRemoteProvider

    GS = GSRemoteProvider()
    os.environ[
        'GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
    print(workflow.default_remote_prefix)

    indir = config["indir"]  #files_path is a GCP directory: gs://mouse_pfc/allc
    if indir.startswith('gs://'):
        bucket_name = indir.replace('gs://','').split('/')[0]
        indir = '/'.join(indir.replace('gs://','').split('/')[1:])
        files = GS.client.list_blobs(bucket_name,prefix=indir)
        input_files = [file.name for file in files if file.name.endswith('.gz') or file.name.endswith('.mz')]
    else:
        files_path = config["files_path"]
        print(f"files_path: {files_path}")
        input_files = pd.read_csv(os.path.expanduser(files_path),sep='\t',
            header=None).iloc[:, 0].tolist()

    if '/' in workflow.default_remote_prefix:
        bucket_name = workflow.default_remote_prefix.split('/')[0]
        prefix = '/'.join(workflow.default_remote_prefix.split('/')[1:]) + '/' + outdir
    else:
        bucket_name = workflow.default_remote_prefix
        prefix = outdir
    outfiles = GS.client.list_blobs(bucket_name,prefix=prefix)
    existed_outfiles = [os.path.basename(file.name) for file in outfiles]
else:  #run on local
    config['gcp'] = False
    existed_outfiles = []
    if 'indir' in config:
        indir = config["indir"]
        input_files = [file for file in os.listdir(indir)]
    else:
        files_path = config["files_path"]
        print(f"files_path: {files_path}")
        input_files = pd.read_csv(os.path.expanduser(files_path),sep='\t',
            header=None).iloc[:, 0].tolist()
        indir = os.path.dirname(input_files[0])

if 'suffix' in config:
    suffix = config["suffix"]
else:
    # suffix = ".allc.tsv.gz"
    suffix = '.' + '.'.join(os.path.basename(input_files[0]).split('.')[1:])

if not os.path.exists(outdir) and ('gcp' not in config or ('gcp' in config and not config["gcp"])):
    os.mkdir(outdir)

snames = [os.path.basename(file.replace(suffix,'')) for file in input_files]

if 'chrom' in config:
    chrom = config['chrom']
    if 'chrom_prefix' in config:
        chrom_prefix = config['chrom_prefix']
        os.system(f"gsutil cp -n {chrom_prefix}/{chrom} ./")
else:
    print("When using allcools extract, chrom must be provided")
    chrom = ""

if 'algorithm' not in config:
    out_ext = ".CGN-Merge.allc.tsv.gz"
elif config['algorithm'] == 'allcools':
    out_ext = ".CGN-Merge.allc.tsv.gz"
else:
    out_ext = ".CGN.merged.mz"
    if 'bmi' not in config:  #mm10_with_chrL.allc.mz.CGN.bmi
        raise ValueError("bmi must be provided for bmzip extractCG")
    bmi = config['bmi']
    try:
        bmi_prefix = config['bmi_prefix']
        os.system(f"gsutil cp -n {bmi_prefix}/{bmi} ./")
    except:
        pass

snames = [sname for sname in snames if sname + out_ext not in existed_outfiles]

print(indir,outdir,suffix)
n = len(snames)
print("No of allc files:",str(n))

# for log
if 'files_path' in config:
    write_allc_path = config['files_path']
else:
    write_allc_path = ''

data = f"""
indir: {indir}
outdir: {outdir}
suffix: {suffix}
No. of allc files: {n}
files_path: {write_allc_path}
"""
with open("parameters.txt",'w') as f:
    f.write(data)

rule target_all:
    input:  #[sname+'.mz' for sname in snames]
        expand(outdir + "/{sample}" + out_ext,sample=snames)

rule allcools_extract:
    input:
        allc_file=lambda wildcards: os.path.join(indir,wildcards.sname + suffix),
        allc_file_idx=lambda wildcards: os.path.join(indir,wildcards.sname + suffix + '.tbi')

    output:
        allc_file="{outdir}/{sname}.CGN-Merge.allc.tsv.gz",
        allc_file_idx="{outdir}/{sname}.CGN-Merge.allc.tsv.gz.tbi"

    params:
        chrom=chrom,
        prefix=lambda wildcards: os.path.join(wildcards.outdir,wildcards.sname) if not config['gcp'] else \
            os.path.join(workflow.default_remote_prefix,wildcards.outdir,wildcards.sname)

    conda:
        "yap"

    shell:
        """
        allcools extract --strandness merge --output_format allc \
        --files_path {input.allc_file} \
        --output_prefix {params.prefix} \
        --mc_contexts CGN --chrom_size_path {params.chrom}
        """

rule bmzip_extract:
    input:
        mz_file=lambda wildcards: os.path.join(indir,wildcards.sname + suffix)

    output:
        mz_file="{outdir}/{sname}.CGN.merged.mz",

    params:
        bmi=bmi

    conda:
        "yap"

    shell:
        """
        bmzip extractCG -i {input.mz_file} -o {output.mz_file} -b {params.bmi}
        """

    # allcools:
    # snakemake --use-conda --printshellcmds -s allc2mz.smk --config {indir} {outdir} {allc_path}".0$SKYPILOT_NODE_RANK" {reference} {ref_prefix} {chrom} {chrom_prefix} {gcp} --default-remote-prefix {bucket} --default-remote-provider GS --google-lifesciences-region us-west1 --scheduler greedy -j {cpu}
    # bmzip
    # snakemake --use-conda --printshellcmds -s ~/MySnakemake/workflow/rules/extractCG.smk --config algorithm="bmzip" indir=test_mz files_path=mz.path".0$SKYPILOT_NODE_RANK" outdir=pfc_mz-CGN bmi=mm10_with_chrL.allc.mz.CGN.bmi bmi_prefix=gs://wubin_ref/mm10/annotations gcp=True --default-remote-prefix mouse_pfc --default-remote-provider GS --google-lifesciences-region us-west1 --scheduler greedy -j 4
