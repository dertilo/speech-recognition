from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import os
import shutil

import torch
from pathlib import Path
from pytorch_lightning import LightningDataModule
from time import time
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Union, Optional
from tqdm import tqdm
from util import data_io
from util.util_methods import process_with_threadpool, merge_dicts
from data_related.audio_feature_extraction import (
    get_length,
    AudioFeaturesConfig,
    Sample,
)
from data_related.char_stt_dataset import DataConfig, CharSTTDataset
from lightning.lightning_model import collate
from utils import HOME, BLANK_SYMBOL, SPACE


def download_librispeech_en(
    data_folder,
    base_url="http://www.openslr.org/resources/12",
    files=[
        "train-clean-100.tar.gz",
        "train-clean-360.tar.gz",
        "train-other-500.tar.gz",
        "dev-clean.tar.gz",
        "dev-other.tar.gz",
        "test-clean.tar.gz",
        "test-other.tar.gz",
    ],
):
    for file_name in files:
        data_io.download_data(
            base_url,
            file_name,
            data_folder,
            unzip_it=True,
            verbose=True,
        )
        split_name = file_name.split(".")[0]
        deeper_foler = f"{data_folder}/{split_name}/LibriSpeech/{split_name}"
        if os.path.isdir(deeper_foler):
            datasplit_folder = f"{data_folder}/{split_name}"
            if not os.path.isfile(f"{data_folder}/LICENSE.TXT"):
                for f in list(Path(deeper_foler).rglob("*.TXT")):
                    shutil.move(str(f),datasplit_folder)

            tmp_folder = f"{data_folder}/tmp"
            shutil.move(deeper_foler,tmp_folder)
            shutil.rmtree(datasplit_folder)
            shutil.move(tmp_folder,datasplit_folder)

def read_librispeech(librispeech_folder: str) -> Dict[str, str]:
    """:return dictionary where keys are filenames and values are utterances"""
    p = Path(librispeech_folder)
    audio_files = list(p.rglob("*.flac"))
    print("in %s found %d audio-files" % (librispeech_folder, len(audio_files)))

    def parse_line(l):
        s = l.split(" ")
        return s[0], " ".join(s[1:])

    g = (
        parse_line(l)
        for f in p.rglob("*.trans.txt")
        for l in data_io.read_lines(str(f))
    )
    key2utt = {k: v for k, v in g}

    def build_key(f):
        return str(f).split("/")[-1].replace(".flac", "")

    g = ((f, build_key(f)) for f in audio_files)
    file2utt = {str(f): key2utt[k] for f, k in g if k in key2utt.keys()}
    return file2utt


def load_samples(file: str, base_path: str) -> List[Sample]:
    def adjust_file_path(d):
        d["audio_file"] = os.path.join(base_path, d["audio_file"])
        return Sample(**d)

    return [adjust_file_path(d) for d in data_io.read_jsonl(file)]


def build_samples(folders, raw_data_path):
    corpus = merge_dicts(
        [read_librispeech(os.path.join(raw_data_path, f)) for f in folders]
    )
    assert len(corpus) > 0

    def build_sample(audio_file, text):
        return Sample(
            audio_file.replace(raw_data_path + "/", ""),
            text,
            get_length(audio_file),
        )

    yield from process_with_threadpool(
        # TODO(tilo): still not sure whether this is necessary
        [{"audio_file": f, "text": t} for f, t in corpus.items()],
        build_sample,
        max_workers=10,
    )


def build_librispeech_corpus(
    raw_data_path: str,
    name: str,
    folders: List[str] = None,
    reprocess=False,
) -> List[Sample]:
    file = raw_data_path + "/%s_samples.jsonl.gz" % name

    if not os.path.isfile(file) or reprocess:
        print("preprocessing samples from file: %s" % file)
        samples_to_dump = build_samples(folders, raw_data_path)
        data_io.write_jsonl(file, tqdm(s._asdict() for s in samples_to_dump))

    return load_samples(file, raw_data_path)


def build_dataset(raw_data_path, name="debug", files=["dev-clean"]) -> CharSTTDataset:
    conf = DataConfig(LIBRI_VOCAB)
    audio_conf = AudioFeaturesConfig(feature_type="stft")
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


def debug_methods():
    download_librispeech_en(
        os.environ["HOME"] + "/data/asr_data/ENGLISH",
        files=["dev-clean.tar.gz", "dev-other.tar.gz"],
    )
    datasets = [
        # ("train", ["train-clean-100", "train-clean-360", "train-other-500"]),
        ("eval", ["dev-clean", "dev-other"]),
        # ("test", ["test-clean", "test-other"]),
    ]
    start = time()
    for name, folders in datasets:
        samples = build_librispeech_corpus(
            HOME + "/data/asr_data/ENGLISH/LibriSpeech", name, folders, reprocess=True
        )
        print("%s got %d samples" % (name, len(samples)))

    print("took: %0.2f seconds" % (time() - start))

    """:return
    in %s/train-clean-100 found 28539 audio-files
    in .../train-clean-360 found 104014 audio-files
    in .../train-other-500 found 148688 audio-files
    281241it [01:47, 2614.41it/s]
    train got 281241 samples
    in .../dev-clean found 2703 audio-files
    in .../dev-other found 2864 audio-files
    5567it [00:02, 2689.19it/s]
    eval got 5567 samples
    in .../test-clean found 2620 audio-files
    in .../test-other found 2939 audio-files
    5559it [00:02, 2710.73it/s]
    test got 5559 samples
    took: 127.06 seconds
    """


if __name__ == "__main__":
    download_librispeech_en(
        data_folder="/media/dertilo/daten/data/asr_data/ENGLISH/LibriSpeech",
        # files=["dev-clean.tar.gz", "dev-other.tar.gz"],
    )

    # ldm = LibrispeechDataModule(
    #     os.environ["HOME"] + "/data/asr_data/ENGLISH/LibriSpeech",
    #     collate_fn=collate,
    #     hparams=argparse.Namespace(**{"num_workers": 0, "batch_size": 8}),
    # )
    # ldm.prepare_data()
    # for batch in ldm.train_dataloader():
    #     break
