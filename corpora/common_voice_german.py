from time import time

from pathlib import Path

from typing import Dict

import os

from corpora.common import maybe_extract, AudioConfig, process_write_manifest
from corpora.common_voice import build_audiofile2text

if __name__ == "__main__":
    audio_config = AudioConfig("mp3")

    ac = f"{audio_config.format}{'' if audio_config.bitrate is None else '_' + str(audio_config.bitrate)}"
    corpus_name = "CV_GERMAN"
    base_path = os.environ["HOME"]
    raw_zipfile = f"{base_path}/data/de.tar.gz"
    work_dir = f"{base_path}/data"
    corpus_dir = f"{work_dir}/{corpus_name}"
    raw_dir = f"{corpus_dir}/raw"
    maybe_extract(raw_zipfile, raw_dir)

    for split_name in ["train", "dev", "test"]:
        processed_corpus_dir = f"{corpus_dir}/{split_name}_processed_{ac}"
        file2utt = build_audiofile2text(raw_dir, split_name, "de")
        print(f"beginn processing {processed_corpus_dir}")
        start = time()
        process_write_manifest((raw_dir, processed_corpus_dir), file2utt, audio_config)
        print(f"processing done in: {time() - start} secs")

"""
beginn processing /home/thimmelsbach/data/CV_GERMAN/train_processed_mp3
246525it [1:11:48, 57.21it/s]   
beginn processing /home/thimmelsbach/data/CV_GERMAN/dev_processed_mp3
15588it [04:58, 52.19it/s] 
beginn processing /home/thimmelsbach/data/CV_GERMAN/test_processed_mp3
15588it [04:48, 54.08it/s]
"""