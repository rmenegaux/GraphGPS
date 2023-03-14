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

    main="python /gpfswork/rech/tbr/ump88gx/EJ_GraphGPS/GraphGPS/main.py --cfg ${cfg_file}"
    out_dir="results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run each repeat as a separate job
    for SEED in {0..3}; do
        script="sbatch -J ${cfg_suffix}-${dataset} --error=/gpfswork/rech/tbr/ump88gx/EJ_logs/%j.out --output=/gpfswork/rech/tbr/ump88gx/EJ_logs/%j.out run/wrapper.sb ${main} --repeat 1 seed ${SEED} ${common_params}"
        echo $script
        eval $script
    done
    # script="sbatch -J ${cfg_suffix}-${dataset} --error=/gpfswork/rech/tbr/ump88gx/EJ_logs/%j.out --output=/gpfswork/rech/tbr/ump88gx/EJ_logs/%j.out run/wrapper.sb ${main} --repeat 1 train.auto_resume True ${common_params}"
    # echo $script
    # eval $script
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
main_dir="/gpfswork/rech/tbr/ump88gx/EJ_GraphGPS/GraphGPS"
cfg_dir="${main_dir}/configs/ZINC"

DATASET="zinc"
cfg_suffix="GraphiT_EJ_QK+E_mono_V+E_multi_HeadsOnly"
name="'QK+E_mono_V+E_multi_DptConn_HeadsOnly_4seeds'"
#addition="'addition'"
#multiplication="'multiplication'"
#QK_op="'QK_op'"
#KE_op="'KE_op'"
#VE_op="'VE_op'"
#dropout_lvl="'dropout_lvl'"
#connections="'connections'"
#edge_out_dim="'edge_out_dim'"
args="name_tag ${name} wandb.mode offline"
#dataset.dir ${main_dir}/datasets/ZINC n_heads 1 wandb.name ${name} gt.layer_args '[{${QK_op}:${addition}}, {${KE_op}:${addition}}, {${VE_op}:${multiplication}}, {${dropout_lvl}:${connections}}, {${edge_out_dim}:null}]'"

run_repeats ${DATASET} ${cfg_suffix} "${args}"


# DATASET="mnist"
# run_repeats ${DATASET} GPS "name_tag GPSwLapPE.GatedGCN+Trf.10run"


# DATASET="cifar10"
# run_repeats ${DATASET} GPS "name_tag GPSwLapPE.GatedGCN+Trf.10run"


# DATASET="pattern"
# run_repeats ${DATASET} GPS "name_tag GPSwLapPE.GatedGCN+Trf.eigv16.lr0005"


# DATASET="cluster"
# run_repeats ${DATASET} GPS "name_tag GPSwLapPE.GatedGCN+Trf.lr0005.10run"


# DATASET="ogbg-molhiv"
# run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.GatedGCN+Trf.lyr10.wd-5.drp005.10run"


# DATASET="ogbg-molpcba"
# run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.dim384.meanpool.wBNposmlp1.wd-5.10runs"


# DATASET="ogbg-code2"
# run_repeats ${DATASET} GPS "name_tag GPSnoPE.GatedGCN+Perf.drp02.wd-5"


# DATASET="ogbg-ppa"  # NOTE: for ogbg-ppa we need SBATCH --mem=48G
# run_repeats ${DATASET} GPS "name_tag GPSnoPE.GatedGCN+Perf.lyr3.dim256.drp01.wd-5.10run"


# DATASET="pcqm4m"  # NOTE: for PCQM4Mv2 we need SBATCH --mem=48G and up to 3days runtime; run only one repeat
# run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.small.lyr5.dim304"
# run_repeats ${DATASET} GPSmedium+RWSE "name_tag GPSwRWSE.medium.lyr10.dim384.heads16.drp01.attndrp01.lr0002.e150"


# DATASET="malnettiny"
# run_repeats ${DATASET} GPS-noPE  "name_tag GPS-noPE.GatedGCN+Perf.lyr5.dim64.10runs"
# run_repeats ${DATASET} GPS-noPE  "name_tag GPS-noPE.GatedGCN+Trf.lyr5.dim64.bs4.bacc4.10run  train.batch_size 4 optim.batch_accumulation 4 gt.layer_type CustomGatedGCN+Transformer"


################################################################################
##### extra
################################################################################
cfg_dir="configs/SAN"
DATASET="pattern"
#run_repeats ${DATASET} SAN "name_tag SAN.LapPE.10run"
