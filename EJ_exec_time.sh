conda activate graphgps_
source gpu_setVisibleDevices.sh
GPUID=0
cd /home/ejehanno/repos/GraphGPS

python main.py --cfg configs/LRGB/pcqm4m_subset-GraphiT+RWSE.yaml