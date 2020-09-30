import torch
from argparse import ArgumentParser

from pytorch_lightning import EvalResult, TrainResult
from torch.utils.data import DataLoader
from typing import Union, List, Any, Optional

import pytorch_lightning as pl
import numpy as np

from espnet_lightning.espnet_dataloader import RawSampler, build_sequence_iter_factory


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

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self._dataloader("valid")
