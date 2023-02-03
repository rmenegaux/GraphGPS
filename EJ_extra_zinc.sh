conda activate graphgps_
source gpu_setVisibleDevices.sh
GPUID=0
cd /home/ejehanno/repos/GraphGPS

python main.py --cfg configs/GPS/zinc-GraphiT_EJ_tests.yaml train.auto_resume True wandb.use False

# zinc-GraphiT_EJ_QK+E_mono_V+E_multi.yaml
# zinc-GraphiT_EJ_QK+E_mono_V.yaml
# zinc-GraphiT_EJ_QK+E_multi_V+E_multi_AdjMask_noRings.yaml
# zinc-GraphiT_EJ_QK+E_multi_V+E_multi_AdjMask.yaml
# zinc-GraphiT_EJ_Q+K+E_multi_V+E_multi_noHeads_DoubleScaling.yaml
# zinc-GraphiT_EJ_Q+K+E_multi_VE_multi_noHeads_DoubleScaling.yaml
# zinc-GraphiT_EJ_Q+K+E_multi_V+E_multi_noHeads.yaml
# zinc-GraphiT_EJ_E_multi_V+E_multi.yaml
# zinc-GraphiT_EJ_GINE+GraphiT.yaml