import argparse
import logging
import sys
import torch
from distutils.version import LooseVersion
from espnet.utils.cli_utils import get_commandline_args
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.collect_stats import collect_stats
from espnet2.tasks.abs_task import scheduler_classes, GradScaler
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.load_pretrained_model import load_pretrained_model
from espnet2.torch_utils.model_summary import model_summary
from espnet2.torch_utils.pytorch_version import pytorch_cudnn_version
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption, resolve_distributed_mode
from espnet2.train.reporter import Reporter
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump
from pathlib import Path
from typeguard import check_argument_types

from espnet_lightning.espnet_dataloader import build_sequence_iter_factory
from espnet_lightning.trainer import Trainer

cls = ASRTask


def espnet_asr_train_validate(args: argparse.Namespace):
    (
        distributed_option,
        model,
        optimizers,
        output_dir,
        reporter,
        scaler,
        schedulers,
    ) = setup(args)

    train_validate(
        args,
        distributed_option,
        model,
        optimizers,
        output_dir,
        reporter,
        scaler,
        schedulers,
    )


def espnet_collect_stats(args):
    model = build_model(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Perform on collect_stats mode. This mode has two roles
    # - Derive the length and dimension of all input data
    # - Accumulate feats, square values, and the length for whitening
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size
    if len(args.train_shape_file) != 0:
        train_key_file = args.train_shape_file[0]
    else:
        train_key_file = None
    if len(args.valid_shape_file) != 0:
        valid_key_file = args.valid_shape_file[0]
    else:
        valid_key_file = None
    collect_stats(
        model=model,
        train_iter=cls.build_streaming_iterator(
            data_path_and_name_and_type=args.train_data_path_and_name_and_type,
            key_file=train_key_file,
            batch_size=args.batch_size,
            dtype=args.train_dtype,
            num_workers=args.num_workers,
            allow_variable_data_keys=args.allow_variable_data_keys,
            ngpu=args.ngpu,
            preprocess_fn=cls.build_preprocess_fn(args, train=False),
            collate_fn=cls.build_collate_fn(args, train=False),
        ),
        valid_iter=cls.build_streaming_iterator(
            data_path_and_name_and_type=args.valid_data_path_and_name_and_type,
            key_file=valid_key_file,
            batch_size=args.valid_batch_size,
            dtype=args.train_dtype,
            num_workers=args.num_workers,
            allow_variable_data_keys=args.allow_variable_data_keys,
            ngpu=args.ngpu,
            preprocess_fn=cls.build_preprocess_fn(args, train=False),
            collate_fn=cls.build_collate_fn(args, train=False),
        ),
        output_dir=output_dir,
        ngpu=args.ngpu,
        log_interval=args.log_interval,
        write_collected_feats=args.write_collected_feats,
    )


def train_validate(
    args,
    distributed_option,
    model,
    optimizers,
    output_dir,
    reporter,
    scaler,
    schedulers,
):
    train_iter_factory = build_sequence_iter_factory(
        args=args,
        distributed_option=distributed_option,
        mode="train",
    )
    valid_iter_factory = build_sequence_iter_factory(
        args=args,
        distributed_option=distributed_option,
        mode="valid",
    )
    # 9. Start training
    # Don't give args to trainer.run() directly!!!
    # Instead of it, define "Options" object and build here.
    trainer_options = cls.trainer.build_options(args)
    if isinstance(args.keep_nbest_models, int):
        keep_nbest_models = args.keep_nbest_models
    else:
        if len(args.keep_nbest_models) == 0:
            logging.warning("No keep_nbest_models is given. Change to [1]")
            args.keep_nbest_models = [1]
        keep_nbest_models = max(args.keep_nbest_models)
    Trainer.run(
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        train_iter_factory=train_iter_factory,
        valid_iter_factory=valid_iter_factory,
        plot_attention_iter_factory=None,
        reporter=reporter,
        scaler=scaler,
        output_dir=output_dir,
        max_epoch=args.max_epoch,
        seed=args.seed,
        patience=args.patience,
        keep_nbest_models=keep_nbest_models,
        early_stopping_criterion=args.early_stopping_criterion,
        best_model_criterion=args.best_model_criterion,
        val_scheduler_criterion=args.val_scheduler_criterion,
        trainer_options=trainer_options,
        distributed_option=distributed_option,
    )
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        # Generated n-best averaged model
        average_nbest_models(
            reporter=reporter,
            output_dir=output_dir,
            best_model_criterion=args.best_model_criterion,
            nbest=args.keep_nbest_models,
        )


def setup(args):
    # -------------------AbsTask.main--------------------------------------------
    print(get_commandline_args(), file=sys.stderr)
    # "distributed" is decided using the other command args
    resolve_distributed_mode(args)
    # ------------------------------------------------------------------------------
    assert check_argument_types()
    # 0. Init distributed process
    distributed_option = build_dataclass(DistributedOption, args)
    distributed_option.init()
    _rank = ""
    # 1. Set random-seed
    set_all_random_seed(args.seed)
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    model = build_model(args)
    # 3. Build optimizer
    optimizers = cls.build_optimizers(args, model=model)
    # 4. Build schedulers
    schedulers = []
    for i, optim in enumerate(optimizers, 1):
        suf = "" if i == 1 else str(i)
        name = getattr(args, f"scheduler{suf}")
        conf = getattr(args, f"scheduler{suf}_conf")
        if name is not None:
            cls_ = scheduler_classes.get(name)
            if cls_ is None:
                raise ValueError(f"must be one of {list(scheduler_classes)}: {name}")
            scheduler = cls_(optim, **conf)
        else:
            scheduler = None

        schedulers.append(scheduler)
    logging.info(pytorch_cudnn_version())
    logging.info(model_summary(model))
    for i, (o, s) in enumerate(zip(optimizers, schedulers), 1):
        suf = "" if i == 1 else str(i)
        logging.info(f"Optimizer{suf}:\n{o}")
        logging.info(f"Scheduler{suf}: {s}")
    # 5. Dump "args" to config.yaml
    # NOTE(kamo): "args" should be saved after object-buildings are done
    #  because they are allowed to modify "args".
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        logging.info(f'Saving the configuration in {output_dir / "config.yaml"}')
        yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)
    # 6. Loads pre-trained model
    for p, k in zip(args.pretrain_path, args.pretrain_key):
        load_pretrained_model(
            model=model,
            # Directly specify the model path e.g. exp/train/loss.best.pt
            pretrain_path=p,
            # if pretrain_key is None -> model
            # elif pretrain_key is str e.g. "encoder" -> model.encoder
            pretrain_key=k,
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            map_location=f"cuda:{torch.cuda.current_device()}"
            if args.ngpu > 0
            else "cpu",
        )
    # 7. Resume the training state from the previous epoch
    reporter = Reporter()
    if args.use_amp:
        if LooseVersion(torch.__version__) < LooseVersion("1.6.0"):
            raise RuntimeError("Require torch>=1.6.0 for  Automatic Mixed Precision")
        scaler = GradScaler()
    else:
        scaler = None
    if args.resume and (output_dir / "checkpoint.pth").exists():
        cls.resume(
            checkpoint=output_dir / "checkpoint.pth",
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            reporter=reporter,
            scaler=scaler,
            ngpu=args.ngpu,
        )
    return (
        distributed_option,
        model,
        optimizers,
        output_dir,
        reporter,
        scaler,
        schedulers,
    )


def build_model(args):
    model = cls.build_model(args=args)
    if not isinstance(model, AbsESPnetModel):
        raise RuntimeError(
            f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
        )
    model = model.to(
        dtype=getattr(torch, args.train_dtype),
        device="cuda" if args.ngpu > 0 else "cpu",
    )
    return model
