from espnet2.utils.build_dataclass import build_dataclass
from typing import Union, Sequence, Any

import logging

import argparse

import numpy as np
import torch
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.build_batch_sampler import build_batch_sampler
from espnet2.tasks.abs_task import IteratorOptions
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.distributed_utils import DistributedOption, resolve_distributed_mode
from espnet2.train.preprocessor import CommonPreprocessor
from torch.utils.data import DataLoader
from typeguard import check_return_type



class RawSampler(AbsSampler):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def generate(self, seed):
        return list(self.batches)


class SequenceIterFactory(AbsIterFactory):
    """Build iterator for each epoch.

    This class simply creates pytorch DataLoader except for the following points:
    - The random seed is decided according to the number of epochs. This feature
      guarantees reproducibility when resuming from middle of training process.
    - Enable to restrict the number of samples for one epoch. This features
      controls the interval number between training and evaluation.

    """

    def __init__(
        self,
        dataset,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        num_iters_per_epoch: int = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ):

        if not isinstance(batches, AbsSampler):
            self.sampler = RawSampler(batches)
        else:
            self.sampler = batches

        self.dataset = dataset
        self.num_iters_per_epoch = num_iters_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory
        self._epoch = 0

    def build_iter(self) -> DataLoader:

        shuffle = self.shuffle
        assert self.num_iters_per_epoch is None

        batches = self.sampler.generate(self.seed)
        if shuffle:
            np.random.RandomState(self._epoch + self.seed).shuffle(batches)
            self._epoch +=1

        # For backward compatibility for pytorch DataLoader
        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batches,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **kwargs,
        )

def build_preprocess_fn(
        args: argparse.Namespace, train: bool
):
    if args.use_preprocessor:
        retval = CommonPreprocessor(
            train=train,
            token_type=args.token_type,
            token_list=args.token_list,
            bpemodel=args.bpemodel,
            non_linguistic_symbols=args.non_linguistic_symbols,
            text_cleaner=args.cleaner,
            g2p_type=args.g2p,
        )
    else:
        retval = None
    assert check_return_type(retval)
    return retval


def build_collate_fn(
        args: argparse.Namespace, train: bool
) :
    return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)


def build_iter_options(
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
):
    if mode == "train":
        preprocess_fn = build_preprocess_fn(args, train=True)
        collate_fn = build_collate_fn(args, train=True)
        data_path_and_name_and_type = args.train_data_path_and_name_and_type
        shape_files = args.train_shape_file
        batch_size = args.batch_size
        batch_bins = args.batch_bins
        batch_type = args.batch_type
        max_cache_size = args.max_cache_size
        distributed = distributed_option.distributed
        num_batches = None
        num_iters_per_epoch = args.num_iters_per_epoch
        train = True

    elif mode == "valid":
        preprocess_fn = build_preprocess_fn(args, train=False)
        collate_fn = build_collate_fn(args, train=False)
        data_path_and_name_and_type = args.valid_data_path_and_name_and_type
        shape_files = args.valid_shape_file

        if args.valid_batch_type is None:
            batch_type = args.batch_type
        else:
            batch_type = args.valid_batch_type
        if args.valid_batch_size is None:
            batch_size = args.batch_size
        else:
            batch_size = args.valid_batch_size
        if args.valid_batch_bins is None:
            batch_bins = args.batch_bins
        else:
            batch_bins = args.valid_batch_bins
        if args.valid_max_cache_size is None:
            # Cache 5% of maximum size for validation loader
            max_cache_size = 0.05 * args.max_cache_size
        else:
            max_cache_size = args.valid_max_cache_size
        distributed = distributed_option.distributed
        num_batches = None
        num_iters_per_epoch = None
        train = False

    elif mode == "plot_att":
        preprocess_fn = build_preprocess_fn(args, train=False)
        collate_fn = build_collate_fn(args, train=False)
        data_path_and_name_and_type = args.valid_data_path_and_name_and_type
        shape_files = args.valid_shape_file
        batch_type = "unsorted"
        batch_size = 1
        batch_bins = 0
        num_batches = args.num_att_plot
        # num_att_plot should be a few sample ~ 3, so cache all data.
        max_cache_size = np.inf if args.max_cache_size != 0.0 else 0.0
        # always False because plot_attention performs on RANK0
        distributed = False
        num_iters_per_epoch = None
        train = False
    else:
        raise NotImplementedError(f"mode={mode}")

    return IteratorOptions(
        preprocess_fn=preprocess_fn,
        collate_fn=collate_fn,
        data_path_and_name_and_type=data_path_and_name_and_type,
        shape_files=shape_files,
        batch_type=batch_type,
        batch_size=batch_size,
        batch_bins=batch_bins,
        num_batches=num_batches,
        max_cache_size=max_cache_size,
        distributed=distributed,
        num_iters_per_epoch=num_iters_per_epoch,
        train=train,
    )

def build_sequence_iter_factory(
    args: argparse.Namespace, mode: str,
):
    resolve_distributed_mode(args)
    distributed_option = build_dataclass(DistributedOption, args)
    distributed_option.init()

    iter_options = build_iter_options(args, distributed_option, mode)


    dataset = ESPnetDataset(
        iter_options.data_path_and_name_and_type,
        float_dtype=args.train_dtype,
        preprocess=iter_options.preprocess_fn,
        max_cache_size=iter_options.max_cache_size,
    )

    batch_sampler = build_batch_sampler(
        type=iter_options.batch_type,
        shape_files=iter_options.shape_files,
        fold_lengths=args.fold_length,
        batch_size=iter_options.batch_size,
        batch_bins=iter_options.batch_bins,
        sort_in_batch=args.sort_in_batch,
        sort_batch=args.sort_batch,
        drop_last=False,
        min_batch_size=torch.distributed.get_world_size()
        if iter_options.distributed
        else 1,
    )

    batches = list(batch_sampler)
    if iter_options.num_batches is not None:
        batches = batches[: iter_options.num_batches]

    bs_list = [len(batch) for batch in batches]

    logging.info(f"[{mode}] dataset:\n{dataset}")
    logging.info(f"[{mode}] Batch sampler: {batch_sampler}")
    logging.info(
        f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
        f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, max={np.max(bs_list)}"
    )

    if iter_options.distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        for batch in batches:
            if len(batch) < world_size:
                raise RuntimeError(
                    f"The batch-size must be equal or more than world_size: "
                    f"{len(batch)} < {world_size}"
                )
        batches = [batch[rank::world_size] for batch in batches]

    return SequenceIterFactory(
        dataset=dataset,
        batches=batches,
        seed=args.seed,
        num_iters_per_epoch=iter_options.num_iters_per_epoch,
        shuffle=iter_options.train,
        num_workers=args.num_workers,
        collate_fn=iter_options.collate_fn,
        pin_memory=args.ngpu > 0,
    )
