from time import time

from pathlib import Path

from typing import Dict

import os

from util import data_io

from corpora.common import maybe_extract, AudioConfig, process_write_manifest


def common_voice_data(path, split_name: str):
    g = data_io.read_lines(
        os.path.join(path, f"{path}/cv-corpus-6.1-2020-12-11/es/{split_name}.tsv")
    )
    header = next(g).split("\t")

    def parse_line(l):
        d = {k: v for k, v in zip(header, l.split("\t"))}
        return d

    return map(parse_line, g)


def build_audiofile2text(path, split_name) -> Dict[str, str]:
    key2utt = {d["path"]: d["sentence"] for d in common_voice_data(path, split_name)}

    def get_file_name(audio_file):
        return str(audio_file).split("/")[-1]

    malpaudios = [
        "common_voice_es_19499901.mp3",
        "common_voice_es_19499893.mp3",
    ]  # broken audios

    return {
        str(f): key2utt[get_file_name(f)]
        for f in Path(path).rglob("*.mp3")
        if get_file_name(f) in key2utt.keys() and get_file_name(f) not in malpaudios
    }


if __name__ == "__main__":
    audio_config = AudioConfig("mp3")

    ac = f"{audio_config.format}{'' if audio_config.bitrate is None else '_' + str(audio_config.bitrate)}"
    corpus_name = "SPANISH_CV"

    raw_zipfile = "/data/es.tar.gz"
    work_dir = "/data"
    corpus_dir = f"{work_dir}/{corpus_name}"
    raw_dir = f"{corpus_dir}/raw"
    maybe_extract(raw_zipfile, raw_dir)

    for split_name in ["train", "dev", "test"]:
        processed_corpus_dir = f"{corpus_dir}/{split_name}_processed_{ac}"
        file2utt = build_audiofile2text(raw_dir, split_name)
        print(f"beginn processing {processed_corpus_dir}")
        start = time()
        process_write_manifest((raw_dir, processed_corpus_dir), file2utt, audio_config)
        print(f"processing done in: {time() - start} secs")

"""
beginn processing /data/SPANISH_CV/train_processed_mp3
161811it [43:20, 62.22it/s]   
beginn processing /data/SPANISH_CV/dev_processed_mp3
15089it [04:16, 58.78it/s]   
beginn processing /data/SPANISH_CV/test_processed_mp3
15089it [04:16, 58.93it/s]   
"""