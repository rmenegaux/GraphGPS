conda activate graphgps_
source gpu_setVisibleDevices.sh
GPUID=0
cd /home/ejehanno/repos/GraphGPS

python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QK+E_mono)*(V+E_multi)_DptConn" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'addition'}, {'VE_op':'addition'}, {'dropout_lvl':'connections'}, {'edge_out_dim':1}]" && python main.py --cfg configs/GPS/zinc-GINE+GraphiT_EJ_tests.yaml