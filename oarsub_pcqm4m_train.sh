#!/bin/bash
#OAR -l walltime=48:00:00
#OAR -p host='gpuhost15' OR host='gpuhost17' OR host='gpuhost20' OR host='gpuhost28'
source gpu_setVisibleDevices.sh

GPUID=0

#source /home/rmenegau/.bashrc
export PATH="/scratch/curan/rmenegau/miniconda3/bin:$PATH"
# source activate cwn_tensorboard
source activate graphgps
cd /scratch/curan/rmenegau/GraphGPS
# config="configs/GPS/zinc-GPS+RWSE-graphiT.yaml"
# config="configs/GPS/zinc-GINE+RWSE+Rings.yaml"
# config="configs/GPS/zinc-GPS+RWSE-graphiT-Rings-RWSEEdge.yaml"
config="configs/GPS/pcqm4m-GraphiT+RWSE.yaml"
python main.py --cfg $config wandb.use True
# OAR -p host='gpuhost2' OR host='gpuhost15' OR host='gpuhost17'OR host='gpuhost20' OR host='gpuhost28'
