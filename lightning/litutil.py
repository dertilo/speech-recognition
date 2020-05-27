import argparse
import logging
import random
from typing import NamedTuple, Type, Dict

logging.disable(logging.CRITICAL)
import os
import torch
from torch import Tensor
from test_tube import HyperOptArgumentParser
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.mlflow import MLFlowLogger
import numpy as np
import pytorch_lightning as pl


def add_generic_args(parser):
    # fmt: off
    parser.add_argument("--exp_name",default="debug",type=str,help="experiment name")
    parser.add_argument("--run_name", default="test-run", type=str, help="run name")

    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        required=True,
        help="directory where logs and checkpoints will be written",
    )

    parser.add_argument(
        "--fp16",
        default=False,
        type=bool,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    # fmt: on


def setup_mlflowlogger_and_checkpointer(exp_name, save_path):
    mlflow_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri=save_path)
    run_id = mlflow_logger.run_id
    checkpoints_folder = os.path.join(
        save_path, mlflow_logger._expt_id, run_id, "checkpoints"
    )
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=checkpoints_folder, monitor="val_loss", save_top_k=1
    )
    return checkpoint, mlflow_logger, run_id


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def build_args(model_class: Type[pl.LightningModule], override_args: Dict = {}):
    parent_parser = argparse.ArgumentParser(add_help=False)
    add_generic_args(parent_parser)
    parser = model_class.add_model_specific_args(parent_parser)
    override_args = [x for k, v in override_args.items() for x in ["--%s" % k, str(v)]]
    args = parser.parse_args(override_args)
    return args


def generic_train(model: pl.LightningModule, args: argparse.Namespace):
    set_seed(args)

    checkpoint, mlflow_logger, run_id = setup_mlflowlogger_and_checkpointer(
        args.exp_name, args.save_path
    )

    trainer = pl.Trainer(
        logger=mlflow_logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint,
        distributed_backend="ddp" if args.n_gpu > 1 else None,
        use_amp=args.fp16,
        amp_level=args.fp16_opt_level,
        val_check_interval=1.0,
    )

    trainer.fit(model)
    # mlflow_logger.experiment.log_artifacts(run_id, checkpoint.dirpath) # only makes sense if mlflow-loggers artifact-path is not locally but different to save_path
    return trainer
