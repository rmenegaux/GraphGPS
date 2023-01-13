conda activate graphgps_
source gpu_setVisibleDevices.sh
GPUID=0
cd /home/ejehanno/repos/GraphGPS

python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QK+E)*(VE)_multi_DptConn" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'addition'}, {'VE_op':'multiplication'}, {'dropout_lvl':'connections'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QK+E)*(VE)_mono_DptConn" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'addition'}, {'VE_op':'multiplication'}, {'dropout_lvl':'connections'}, {'edge_out_dim':1}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(Q+K+E)*(VE)_multi_DptConn" gt.layer_args "[{'QK_op':'addition'}, {'KE_op':'addition'}, {'VE_op':'multiplication'}, {'dropout_lvl':'connections'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(Q+K+E)*(VE)_multi_DptConn_noHeads" gt.n_heads 1 gt.layer_args "[{'QK_op':'addition'}, {'KE_op':'addition'}, {'VE_op':'multiplication'}, {'dropout_lvl':'connections'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QKE)*(VE)_multi_DptConn" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'multiplication'}, {'VE_op':'multiplication'}, {'dropout_lvl':'connections'}]"