conda activate graphgps_
source gpu_setVisibleDevices.sh
GPUID=0
cd /home/ejehanno/repos/GraphGPS

python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QK+E)*(V+E)_multi_DptNode" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'addition'}, {'VE_op':'addition'}, {'dropout_lvl':'node'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QK+E)*(V+E)_mono_DptNode" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'addition'}, {'VE_op':'addition'}, {'dropout_lvl':'node'}, {'edge_out_dim':1}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QK+E)*(V)_multi_DptNode" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'addition'}, {'dropout_lvl':'node'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QK+E)*(V)_mono_DptNode" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'addition'}, {'dropout_lvl':'node'}, {'edge_out_dim':1}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(Q+K+E)*(V+E)_multi_DptNode" gt.layer_args "[{'QK_op':'addition'}, {'KE_op':'addition'}, {'VE_op':'addition'}, {'dropout_lvl':'node'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(Q+K+E)*(V+E)_multi_DptNode_noHeads" gt.n_heads 1 gt.layer_args "[{'QK_op':'addition'}, {'KE_op':'addition'}, {'VE_op':'addition'}, {'dropout_lvl':'node'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(Q+K+E)*(V)_multi_DptNode" gt.layer_args "[{'QK_op':'addition'}, {'KE_op':'addition'}, {'dropout_lvl':'node'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(Q+K+E)*(V)_multi_DptNode_noHeads" gt.n_heads 1 gt.layer_args "[{'QK_op':'addition'}, {'KE_op':'addition'}, {'dropout_lvl':'node'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QKE)*(V+E)_multi_DptNode" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'multiplication'}, {'VE_op':'addition'}, {'dropout_lvl':'node'}]"
python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml wandb.name "(QKE)*(V)_multi_DptNode" gt.layer_args "[{'QK_op':'multiplication'}, {'KE_op':'multiplication'}, {'dropout_lvl':'node'}]"