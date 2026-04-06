import os.path
import pandas as pd
import os

if 'outdir' in config:
    outdir = config["outdir"]  #
else:
    outdir = 'mz'

if 'gcp' in config and config["gcp"]:
    from snakemake.remote.GS import RemoteProvider as GSRemoteProvider

    GS = GSRemoteProvider()
    os.environ[
        'GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
    print(workflow.default_remote_prefix)

    indir = config["indir"]  #allc_path is a GCP directory: gs://mouse_pfc/allc
    if indir.startswith('gs://'):
        bucket_name = indir.replace('gs://','').split('/')[0]
        indir = '/'.join(indir.replace('gs://','').split('/')[1:])
        files = GS.client.list_blobs(bucket_name,prefix=indir)
        allc_files = [file.name for file in files if file.name.endswith('.gz')]
    else:
        allc_path = config["allc_path"]
        print(f"allc_path: {allc_path}")
        allc_files = pd.read_csv(os.path.expanduser(allc_path),sep='\t',
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
    existed_outfiles = []
    if 'indir' in config:
        indir = config["indir"]
        allc_files = [file for file in os.listdir(indir) if file.endswith('.gz')]
    else:
        allc_path = config["allc_path"]
        print(f"allc_path: {allc_path}")
        allc_files = pd.read_csv(os.path.expanduser(allc_path),sep='\t',
            header=None).iloc[:, 0].tolist()
        indir = os.path.dirname(allc_files[0])
#allc_path is a file with only one column, no header, if allc files are stored on
# cloud, the final path will be "gs://"+workflow.default_remote_prefix+
# path in allc_path file.

if 'suffix' in config:
    suffix = config["suffix"]
else:
    # suffix = ".allc.tsv.gz"
    suffix = '.' + '.'.join(os.path.basename(allc_files[0]).split('.')[1:])

if not os.path.exists(outdir) and ('gcp' not in config or ('gcp' in config and not config["gcp"])):
    os.mkdir(outdir)

snames = [os.path.basename(file.replace(suffix,'')) for file in allc_files]
if 'reference' in config:
    reference = config['reference']
    if 'ref_prefix' in config:
        ref_prefix = config['ref_prefix']
        os.system(f"gsutil cp -n {ref_prefix}/{reference} ./")
else:
    reference = None

if 'chrom' in config:
    chrom = config['chrom']
    if 'chrom_prefix' in config:
        chrom_prefix = config['chrom_prefix']
        os.system(f"gsutil cp -n {chrom_prefix}/{chrom} ./")
else:
    chrom = None

snames = [sname for sname in snames if sname + '.mz' not in existed_outfiles]

print(indir,outdir,suffix,reference)
n = len(snames)
print("No of allc files:",str(n))

# for log
if 'allc_path' in config:
    write_allc_path = config['allc_path']
else:
    write_allc_path = ''

data = f"""
indir: {indir}
outdir: {outdir}
suffix: {suffix}
reference: {reference}
No. of allc files: {n}
allc_path: {write_allc_path}
"""
with open("parameters.txt",'w') as f:
    f.write(data)

rule target_all:
    input:  #[sname+'.mz' for sname in snames]
        expand(outdir + "/{sample}.mz",sample=snames)

rule run_allc2mz:
    input:
        allc_file=lambda wildcards: os.path.join(indir,wildcards.sname + suffix),
        allc_file_idx=lambda wildcards: os.path.join(indir,wildcards.sname + suffix + '.tbi')

    output:
        "{outdir}/{sname}.mz"

    params:
        reference='' if reference is None else f"--reference {reference}",
        chrom='' if chrom is None else f"-P {chrom}"

    shell:
        """
        cytozip allc2cz {input.allc_file} {output} {params.reference} {params.chrom}
        """
