from __future__ import absolute_import, division, print_function, unicode_literals

import os
from typing import List
from util import data_io
from corpora.librispeech import librispeech_corpus
from data_related.audio_util import Sample
from data_related.processing_corpora import process_samples
from utils import HOME


def load_samples(file: str, base_path: str):
    def process(d):
        split_str = 'asr_data/'
        if split_str in d['audio_file']: # TODO(tilo):only for backward compatibility
            _,s = d["audio_file"].split(split_str)
            p,_ = base_path.split(split_str)
            file = os.path.join(p, split_str, s)
        else:
            file = os.path.join(base_path, d["audio_file"])
        d["audio_file"] = file
        return Sample(**d)

    return [process(d) for d in data_io.read_jsonl(file)]


def build_librispeech_corpus(raw_data_path, name: str, folders: List[str]):
    corpus = {
        k: v
        for folder in folders
        for k, v in librispeech_corpus(os.path.join(raw_data_path, folder)).items()
    }

    assert len(corpus) > 0
    file = raw_data_path + "/%s_sorted_samples.jsonl" % name

    if os.path.isfile(file):
        print('loading processed samples from %s'%file)
        samples = load_samples(file, raw_data_path)
    else:
        samples = list(process_samples(corpus))
        data_io.write_jsonl(file, (s._asdict() for s in samples))

    return samples


if __name__ == "__main__":
    # datasets = [
    #     ("train", ["train-clean-100", "train-clean-360", "train-clean-500"]),
    #     ("eval", ["dev-clean", "dev-other"]),
    #     ("test", ["test-clean", "test-other"]),
    # ]
    samples = build_librispeech_corpus(HOME+'/data/asr_data/ENGLISH/LibriSpeech','eval',["dev-clean", "dev-other"])
    print()