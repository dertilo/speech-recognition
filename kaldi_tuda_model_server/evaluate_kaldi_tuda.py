from pprint import pprint

from typing import Dict

import os

from util import data_io

from data_related.utils import ASRSample
from deepspeech_asr.metrics_calculation import calc_wer, calc_num_word_errors

if __name__ == '__main__':

    data_io.read_jsonl("/home/tilo/data/asr_data/GERMAN/results_old_errors.jsonl")

    base_path = f"{os.environ['HOME']}/data/asr_data/GERMAN/tuda"
    folder = "dev_processed_wav"

    asr_samples = (
        ASRSample(**s)
        for s in data_io.read_jsonl(f"{base_path}/{folder}/manifest.jsonl.gz",limit=100)
    )
    id2sample:Dict[str,ASRSample] = {s.audio_file:s for s in asr_samples}

    hyps_refs = [(d["text"],id2sample[d["id"]].text)
     for d in data_io.read_jsonl("/home/tilo/data/asr_data/GERMAN/results_old_errors.jsonl")
                 if "text" in d
                 ]
    pprint([(h,r,calc_num_word_errors(h, r)) for h,r in hyps_refs])
    print(calc_wer(*zip(*hyps_refs)))