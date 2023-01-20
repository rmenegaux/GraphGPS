#!/bin/bash -l
#SBATCH --job-name graph_gps
#SBATCH -A tbr@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=13:00:00
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
config="configs/GPS/zinc-GraphiT+Rings-RWSEfly.yaml"
# config="configs/GPS/ogbg-molpcba-GPS+RWSE.yaml"
# config="configs/GPS/mnist-GraphiT.yaml"
#config="configs/GPS/pcqm4m_subset-GraphiT+RWSE-Atom1fly.yaml"
# config="configs/GPS/pcqm4m_subset-GraphiT+RWSEonthefly.yaml"
#config="configs/GPS/ogbg-molpcba-GraphiT+RWSE.yaml"
# python main.py --cfg $config  wandb.use True wandb.mode "offline" wandb.name "pcba.GraphiT-RWSEonthefly.r0"
python main.py --cfg $config  wandb.use True wandb.mode "offline" wandb.name "zinc.GraphiT-RWSEonthefly-Rings.r0"
