from pathlib import Path

import argparse

import torch
from argparse import ArgumentParser
from espnet2.tasks.asr import ASRTask

from pytorch_lightning import EvalResult, TrainResult, Trainer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Union, List, Any, Optional, Sequence, Dict, Tuple

import pytorch_lightning as pl
import numpy as np

from espnet_lightning.espnet_asr import (
    build_model,
    build_schedulers,
    load_pretrained,
    resume,
)
from espnet_lightning.espnet_dataloader import RawSampler, build_sequence_iter_factory


class LitEspnet(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.model = build_model(args)
        load_pretrained(args.pretrain_path, args.pretrain_key, self.model, args.ngpu)

        # output_dir = Path(args.output_dir) # TODO(tilo)
        # output_dir.mkdir(parents=True, exist_ok=True)
        # if args.resume and (output_dir / "checkpoint.pth").exists():
        #     resume(
        #         checkpoint=output_dir / "checkpoint.pth",
        #         model=self.model,
        #         optimizers=optimizers,
        #         schedulers=schedulers,
        #         reporter=reporter,
        #         scaler=scaler,
        #         ngpu=args.ngpu,
        #     )

    def forward(self, batch):
        return self.model(**batch)

    @staticmethod
    def _log_results(results, loss, prefix, stats):
        stats = {k: v for k, v in stats.items() if v is not None}
        results.log(f"{prefix}loss", loss)
        for k, v in stats.items():
            results.log(f"{prefix}{k}", v, prog_bar=True)

    def training_step(self, ids_batch, batch_idx):
        ids, batch = ids_batch
        loss, stats, weight = self.model(**batch)
        print(stats.keys())
        result = pl.TrainResult(loss)
        self._log_results(result, loss, "train_", stats)
        return result

    def validation_step(self, ids_batch, batch_idx, dataloader_idx=0) -> EvalResult:
        ids, batch = ids_batch
        loss, stats, weight = self.model(**batch)
        result = pl.EvalResult()
        self._log_results(result, loss, "eval_", stats)
        return result

    def configure_optimizers(
        self,
    ) -> Optional[
        Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]
    ]:
        optimizers = ASRTask.build_optimizers(self.args, model=self.model)
        schedulers = build_schedulers(self.args, optimizers)
        return optimizers, schedulers

    #
    # def validation_epoch_end(self, outputs: Union[
    #     EvalResult, List[EvalResult]]) -> EvalResult:
    #     return super().validation_epoch_end(outputs)


class LitEspnetDataModule(pl.LightningDataModule):
    def __init__(
        self, args, train_transforms=None, val_transforms=None, test_transforms=None
    ):
        self.args = args
        super().__init__(train_transforms, val_transforms, test_transforms)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def _dataloader(self, mode: str):
        it_fact = build_sequence_iter_factory(self.args, mode)
        return it_fact.build_iter()

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._dataloader("valid")
