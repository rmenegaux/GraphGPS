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
    for SEED in 0; do
        script="sbatch -J ${cfg_suffix}-${dataset} run/wrapper_large.sb ${main} --repeat 1 seed ${SEED} wandb.use True wandb.mode "offline" ${common_params}"
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

DATASET="pcqm4m"
#run_repeats ${DATASET} "GraphiTmedium+RWSE.lr0002.dropout0.1" "name_tag GraphiTmedium+RWSE.lr0002.dropout0.1 train.auto_resume True train.ckpt_period 2"
#run_repeats ${DATASET} "GraphiTmedium+RWSE.lr0005.dropout0.5" "name_tag GraphiTmedium+RWSE.lr0005.dropout0.5 train.auto_resume True train.ckpt_period 2"
#run_repeats ${DATASET} "GraphiTmedium+RWSE.lr0002.dropout0.5" "name_tag GraphiTmedium+RWSE.lr0002.dropout0.5 train.auto_resume True train.ckpt_period 2"
run_repeats ${DATASET} "GraphiTmedium+RWSE-Rings.lr0002.dropout0.1" "name_tag GraphiTmedium+RWSE-Rings.lr0002.dropout0.1 train.auto_resume True train.ckpt_period 2"
run_repeats ${DATASET} "GraphiT+RWSE-Rings" "name_tag GraphiT+RWSE-Rings train.auto_resume True train.ckpt_period 2"
#run_repeats ${DATASET} "GraphiT+RWSE" "name_tag GraphiT+RWSE train.auto_resume True train.ckpt_period 2"
