#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    dataset=$1
    cfg_suffix=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python main.py --cfg ${cfg_file}"
    out_dir="results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run each repeat as a separate job
    for SEED in {0..3}; do
        script="sbatch -J ${cfg_suffix}-${dataset} run/wrapper.sb ${main} --repeat 1 seed ${SEED} wandb.use True wandb.mode "offline" ${common_params}"
        echo $script
        eval $script
    done
}


echo "Do you wish to sbatch jobs? Assuming this is the project root dir: `pwd`"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done


################################################################################
##### GPS
################################################################################

# Comment-out runs that you don't want to submit.
cfg_dir="configs/GPS"

DATASET="zinc"
run_repeats ${DATASET} GraphiT-noPE "name_tag GraphiT_nonodePE_Rings.4runs"
run_repeats ${DATASET} GraphiT+Rings-noPE "name_tag GraphiT_nonodePE.4runs"
# run_repeats ${DATASET} GraphiT+RingsNC "name_tag GraphiTwRingsNC.10runs"
# run_repeats ${DATASET} GraphiT+RingsOld "name_tag GraphiTwRingsOld.10runs"
# run_repeats ${DATASET} GraphiT+RWSE "name_tag GraphiTwRWSE.10runs"

