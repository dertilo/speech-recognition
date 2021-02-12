import os
from pathlib import Path
from typing import Dict

from util import data_io


def common_voice_data(path, split_name: str, lang="de"):
    g = data_io.read_lines(
        os.path.join(path, f"{path}/cv-corpus-6.1-2020-12-11/{lang}/{split_name}.tsv")
    )
    header = next(g).split("\t")

    def parse_line(l):
        d = {k: v for k, v in zip(header, l.split("\t"))}
        return d

    return map(parse_line, g)


def build_audiofile2text(path, split_name, lang, broken_files=None) -> Dict[str, str]:
    key2utt = {
        d["path"]: d["sentence"] for d in common_voice_data(path, split_name, lang)
    }

    def get_file_name(audio_file):
        return str(audio_file).split("/")[-1]

    return {
        str(f): key2utt[get_file_name(f)]
        for f in Path(path).rglob("*.mp3")
        if get_file_name(f) in key2utt.keys()
        and (broken_files is None or get_file_name(f) not in broken_files)
    }
