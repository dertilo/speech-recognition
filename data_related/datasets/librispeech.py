from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import os
import shutil

import torch
import torchaudio
from pathlib import Path
from pytorch_lightning import LightningDataModule
from time import time
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Union, Optional
from tqdm import tqdm
from util import data_io
from util.util_methods import process_with_threadpool, merge_dicts, exec_command
from data_related.audio_feature_extraction import (
    get_length,
    AudioFeaturesConfig,
)
from data_related.char_stt_dataset import DataConfig, CharSTTDataset
from data_related.datasets.common import SpeechCorpus, prepare_corpora, \
    find_files_build_audio2text_openslr
from utils import HOME, BLANK_SYMBOL, SPACE


class LibriSpeech(SpeechCorpus):

    def build_audiofile2text(self, path) -> Dict[str, str]:
        audio_suffix = ".flac"

        def parse_line(l):
            s = l.split(" ")
            return s[0]+audio_suffix, " ".join(s[1:])

        return find_files_build_audio2text_openslr(
            path,
            parse_line,
            audio_suffix=audio_suffix,
            transcript_suffix=".trans.txt",
        )

    @staticmethod
    def get_corpora() -> List[SpeechCorpus]:
        base_url = "http://www.openslr.org/resources/12"
        return [
            LibriSpeech(name, f"{base_url}/{name}.tar.gz")
            for name in [
                "train-clean-100",
                "train-clean-360",
                "train-other-500",
                "dev-clean",
                "dev-other",
                "test-clean",
                "test-other",
            ]
        ]


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


# fmt: off
LIBRI_VOCAB = [BLANK_SYMBOL, "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", SPACE]
# fmt: on


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


# parser = argparse.ArgumentParser(description="LibriSpeech Data download")
# parser.add_argument("--dump_dir", required=True, default=None, type=str)
# parser.add_argument("--processed_dir", required=True, default=None, type=str)
# parser.add_argument("--data_sets", nargs="+", default="ALL", type=str)
# args = parser.parse_args()

parser = argparse.ArgumentParser(description="LibriSpeech Data download")
parser.add_argument("--dump_dir", default="/tmp/asr_data/ENGLISH", type=str)
parser.add_argument("--processed_dir", default="/tmp/asr_data/ENGLISH", type=str)
parser.add_argument("--data_sets", nargs="+", default="ALL", type=str)
args = parser.parse_args()

if __name__ == "__main__":

    dump_dir = args.dump_dir
    processed_folder = args.processed_dir

    corpora = LibriSpeech.get_corpora()
    datasets = ["dev-clean"] # args.data_sets
    if len(datasets) > 1 or datasets[0] != "ALL":
        corpora = [c for c in corpora if c.name in datasets]

    prepare_corpora(corpora, dump_dir, processed_folder)
