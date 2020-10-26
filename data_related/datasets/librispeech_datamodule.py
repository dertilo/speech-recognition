from typing import Union, List

import argparse

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_related.datasets.librispeech import build_dataset

raise NotImplementedError # TODO(tilo)
# fmt: off
LIBRI_VOCAB = [BLANK_SYMBOL, "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", SPACE]
# fmt: on

def build_dataset(
    raw_data_path,
    name="debug",
    files=["dev-clean"],
    audio_conf=AudioFeaturesConfig(feature_type="stft"),
) -> CharSTTDataset:
    assert False,"DEPRECATED"#TODO(tilo)
    conf = DataConfig(LIBRI_VOCAB)
    samples = build_librispeech_corpus(
        raw_data_path,
        name,
        files,
    )
    dataset = CharSTTDataset(
        samples,
        conf=conf,
        audio_conf=audio_conf,
    )
    return dataset

class LibrispeechDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        collate_fn,
        hparams: argparse.Namespace,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
    ):
        super().__init__(train_transforms, val_transforms, test_transforms)
        self.hparams = hparams
        self.collate_fn = collate_fn
        self.data_path = data_path
        self.splits = dict(
            [
                ("train", ["train-clean-100"]),  # "train-clean-360", "train-other-500"
                ("eval", ["dev-clean", "dev-other"]),
                ("test", ["test-clean", "test-other"]),
            ]
        )

    def prepare_data(self, *args, **kwargs):
        download_librispeech_en(
            self.data_path,
            files=[f"{f}.tar.gz" for ff in self.splits.values() for f in ff],
        )

    def _dataloader(self, split_name):
        dataset = build_dataset(self.data_path, split_name, self.splits[split_name])
        return DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self._dataloader("eval")