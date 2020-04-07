from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.sparse import (
    csr_matrix,
)  # TODO(tilo): if not imported before torch on HPC-cluster it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

import concurrent.futures
import multiprocessing
import os

from tqdm import tqdm
from typing import Dict
from util import data_io

from corpora.librispeech import librispeech_corpus
from data_related.audio_feature_extraction import get_length
from data_related.char_stt_dataset import Sample


def process_sample(audio_file, text):
    return Sample(audio_file, text, get_length(audio_file))


def dump_sorted_corpus(corpus:Dict,file:str):
    num_cpu = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpu) as executor:
        future_to_sample = [
            executor.submit(process_sample, audio_file, text)
            for audio_file, text in corpus.items()
        ]
        samples = (
            future.result()
            for future in tqdm(concurrent.futures.as_completed(future_to_sample))
        )
        data_io.write_jsonl(
            file, (s._asdict() for s in samples)
        )


if __name__ == "__main__":

    HOME = os.environ["HOME"]
    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"
    datasets = [('train',["train-clean-100", "train-clean-360", "train-clean-500"]),
                ('eval',['dev-clean','dev-other']),
                ('test',['test-clean','test-other']),
                ]
    for name,files in datasets:
        corpus = {
            k: v
            for folder in files
            for k, v in librispeech_corpus(os.path.join(raw_data_path, folder)).items()
        }
        print("got %d samples in corpus" % len(corpus))
        file = raw_data_path+ "/%s_sorted_samples.jsonl" % name
        dump_sorted_corpus(corpus,file)
