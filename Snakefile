# Snakefile

import variograd_utils as vu
configfile: "config.yaml"

DATASET = config["dataset_id"]
OUTDIR = vu.dataset(DATASET).output_dir
SUBJECTS = vu.dataset(DATASET).subj_list
PARCELLATION = config["parcellation"]
SCALE = config["parcellation_scale"]

# include rules
include: "rules/surface_area.smk"
include: "rules/FC_gradients.smk"
include: "rules/Ridge_CV.smk"

# End result
rule all:
    input:
        expand(
            f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G{{n_gradients}}_rs{config['random_state']}.perm_test.csv",
            n_gradients=config["n_gradients"]
            ),
        expand(
            f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.perm_test.csv",
            n_gradients=config["n_gradients"]
            ),
        expand(
            f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/ablation_scores_G1-G{{n_gradients}}.{config['abl_metric']}.rs{config['random_state']}.npy",
            n_gradients=config["n_gradients"]
        )

onsuccess:
    shell("snakemake --rulegraph | dot -Gnodesep=1.5 -Granksep=0.3 -Gbgcolor=transparent -Tpng -o rulegraph.png")
