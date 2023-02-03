import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_agg_dir, set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)
    # TODO: Find a way to do something like that ? Restricting chosen arguments
    return ExtendedSchedulerConfig(**cfg.optim.__dict__)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)
    # TODO: Find a way to do something like that ? Restricting chosen arguments
    setattr(cfg.optim, 'train_mode', cfg.train.mode)
    return ExtendedSchedulerConfig(**cfg.optim.__dict__)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(os.path.join(cfg.out_dir, run_name), cfg.wandb.name)
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        # auto_select_device()
        # FIXME: hardcoding the device to avoid errors
        # ---------------------------------------------
        cfg.device = 'cuda:0'
        use_gpu = True
        gpu_id = 0
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if torch.cuda.is_available() and use_gpu:
            logging.info('cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
            device = torch.device("cuda")
        else:
            logging.info('cuda not available')
            device = torch.device("cpu")
        # ---------------------------------------------
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head
            )
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        from torch_geometric.graphgym.checkpoint import load_ckpt
        start_epoch = load_ckpt(model, optimizer, scheduler,
                  cfg.train.epoch_resume)
        print(f'WEIGHTS LOADED FROM EP {start_epoch}')
        from graphgps.train.custom_train import eval_epoch
        scores, E, E_value, batch = eval_epoch(loggers[2], loaders[2], model, split='test')
        torch.save(scores, 'scores_(QK+E)*(V+E)_multi_DptHead.pt')
        torch.save(E, 'E_(QK+E)*(V+E)_multi_DptHead.pt')
        torch.save(E_value, 'Ev_(QK+E)*(V+E)_multi_DptHead.pt')
        torch.save(batch, 'batch_(QK+E)*(V+E)_multi_DptHead.pt')
        import sys; sys.exit()

        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")

# connections="feature"
# edge_out_dim="edge_out_dim"
# args="name_tag (QK+E_mono)*(V+E_mono)_DptFeat_4seeds wandb.mode 'offline' dataset.dir '/gpfswork/rech/tbr/ump88gx/EJ_GraphGPS/GraphGPS/datasets/ZINC' n_heads 1 wandb.name '(Q+K+E_multi)*(V*E_multi)_DptConn_noHeads' gt.layer_args '[{${QK_op}:${multiplication}}, {${KE_op}:${addition}}, {${VE_op}:${addition}}, {${dropout_lvl}:${connections}}, {${edge_out_dim}:1}]'"

# cfg_dir="/gpfswork/rech/tbr/ump88gx/EJ_GraphGPS/GraphGPS/configs/GPS"

# DATASET="zinc"
# addition="addition"
# multiplication="multiplication"
# QK_op="QK_op"
# KE_op="KE_op"
# VE_op="VE_op"
# dropout_lvl="dropout_lvl"
# connections="connections"
# edge_out_dim="edge_out_dim"
# args="name_tag '(Q+K+E_multi)*(V*E_multi)_DptConn_noHeads_4seeds' wandb.mode 'offline' dataset.dir '/gpfswork/rech/tbr/ump88gx/EJ_GraphGPS/GraphGPS/datasets/ZINC' n_heads 1 wandb.name '(Q+K+E_multi)*(V*E_multi)_DptConn_noHeads' gt.layer_args '[{${QK_op}:${addition}}, {${KE_op}:${addition}}, {${VE_op}:${multiplication}}, {${dropout_lvl}:${connections}}, {${edge_out_dim}:null}]'"
# run_repeats ${DATASET} GraphiT_EJ_tests ${args}

# wandb sync -p "ZINC_seeds" -e "emmanuel-jehanno" /gpfswork/rech/tbr/ump88gx/GraphGPS/wandb/ZINC-subset/wandb/*