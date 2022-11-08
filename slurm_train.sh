#!/bin/bash -l
#SBATCH --job-name graph_gps
#SBATCH -A tbr@v100
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


#config=$1
# config="configs/GPS/zinc-GPS+RWSE-graphiT-Rings-RWSEEdge.yaml"
# config="configs/GPS/ogbg-molhiv-GPS+RWSE.yaml"
config="configs/GPS/zinc-GPS+RWSE-graphiT.yaml"
#config="configs/GPS/zinc-GPS+RWSE-graphiT_Ringsold.yaml"
# config="configs/GPS/ogbg-molhiv-GPS+RWSE+Rings-GraphiT.yaml"
echo "$config"
python main.py --cfg $config wandb.use True wandb.mode "offline" seed 42
