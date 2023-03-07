conda activate graphgps_
source gpu_setVisibleDevices.sh
GPUID=0
cd /home/ejehanno/repos/GraphGPS

python main.py --cfg configs/ZINC/zinc-GraphiT_EJ_QK+E_multi_V+E_multi_shared.yaml