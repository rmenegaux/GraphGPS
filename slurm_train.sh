#!/bin/bash -l
#SBATCH --job-name graph_gps
#SBATCH -A tbr@v100
#SBATCH -C v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/tbr/ump88gx/logs/%j.out      # nom du fichier de sortie
#SBATCH --error=/gpfswork/rech/tbr/ump88gx/logs/%j.out       # nom du fichier d'erreur




module purge

source $HOME/.bashrc
# module load cuda/10.1.2
# module load pytorch-gpu/py3/1.6.0
conda activate $WORK/miniconda3/envs/graphgps


cd $WORK/GraphGPS


args=$1

echo "$args"
# config="configs/GPS/zinc-GraphiT+Rings-RWSEfly.yaml"
config_name="ogbg-molpcba-GraphiT+RWSE"
config_name="pcqm4m_subset-GraphiTmediumGelu+RWSE"
# config_name="pcqm4m_subset-GraphiTGelu+RWSE-shared-Rings"
#config_name="cluster-GraphiT"
config="configs/GraphiT/${config_name}.yaml"
python main.py --cfg $config  wandb.use True wandb.mode "offline" wandb.name "${config_name}-post-pooling"
