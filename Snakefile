# Snakefile

import variograd_utils as vu
configfile: "config.yaml"

DATASET = config["dataset_id"]
SUBJECTS = vu.dataset(DATASET).subj_list
PARCELLATION = config["parcellation"]
SCALE = config["parcellation_scale"]

SCRIPTS = f"{workflow.basedir}/scripts"
OUTDIR = vu.dataset(DATASET).output_dir
OUTSUBDIR = config["output_subdir"].format(
    parcellation=config["parcellation"],
    parcellation_scale=config["parcellation_scale"],
    threshold=int(config["threshold"]*100),
    random_state=config["random_state"]
)

SMALLJOB = config["resource_presets"]["small"]
MEDIUMJOB = config["resource_presets"]["medium"]
LARGEJOB = config["resource_presets"]["large"]

# include rules
include: "rules/surface_area.smk"
include: "rules/FC_gradients.smk"
include: "rules/PLS_CV.smk"
include: "rules/PLS_followup.smk"

# End result
rule all:
    input:
        expand(
            f"{OUTDIR}/{OUTSUBDIR}/PLS/CV_results_G1-G{{n_gradients}}.rs{config['random_state']}.perm_test.csv",
            n_gradients=config["n_gradients"]
            ),
        expand(
            f"{OUTDIR}/{OUTSUBDIR}/PLS/Bootstraps_G1-G{{n_gradients}}.rs{config['random_state']}.h5",
            n_gradients=config["n_gradients"]
            ),
        expand(
            f"{OUTDIR}/{OUTSUBDIR}/PLS/NW_FC_Correlation_G1-G{{n_gradients}}.rs{config['random_state']}.csv",
            n_gradients=config["n_gradients"]
            )


onsuccess:
    shell("snakemake --rulegraph | dot -Gnodesep=1.5 -Granksep=0.3 -Gbgcolor=white -Tpng -o rulegraph.png"),
    shell("git diff --quiet rulegraph.png || (git add rulegraph.png && git commit -m 'Update rulegraph after successful run')")
