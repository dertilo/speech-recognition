from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.sparse import (
    csr_matrix,
)  # TODO(tilo): if not imported before torch on HPC-cluster it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

import concurrent.futures
import multiprocessing
import os

from tqdm import tqdm
from typing import Dict
from data_related.audio_feature_extraction import get_length
from data_related.char_stt_dataset import Sample


def process_sample(audio_file, text):
    return Sample(audio_file, text, get_length(audio_file))


def process_samples(corpus: Dict):
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
        for sample in samples:
            yield sample

