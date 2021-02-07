from __future__ import annotations

import torchaudio
from pathlib import Path

import os

from typing import List, Dict
from util import data_io

from datasets.common import (
    SpeechCorpus,
    find_files_build_audio2text_openslr,
    AudioConfig,
    get_extract_process_zip_data,
)
from data_related.utils import ASRSample
from datasets.spanish_corpora import SpanishDialect, TedxSpanish


class LibriSpeech(SpeechCorpus):
    def build_audiofile2text(self, path) -> Dict[str, str]:
        audio_suffix = ".flac"
        transcript_suffix = ".trans.txt"

        def parse_line(l):
            s = l.split(" ")
            return s[0] + audio_suffix, " ".join(s[1:])

        return find_files_build_audio2text_openslr(
            path,
            parse_line,
            audio_suffix=audio_suffix,
            transcript_suffix=transcript_suffix,
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


class TEDLIUM(SpeechCorpus):
    # TODO(tilo): very hacky!!!
    def __init__(self, name: str, url: str) -> None:
        self.name = name

    def build_audiofile2text(self, path) -> Dict[str, str]:
        audio_suffix = "mp3"
        transcript_suffix = "txt"
        audio_files = list(Path(path).rglob(f"*.{audio_suffix}"))
        assert len(audio_files) > 0
        transcript_files = list(Path(path).rglob(f"*.{transcript_suffix}"))

        key2text = {
            file_name: text
            for t_file in transcript_files
            for file_name, text in (
                (str(t_file).split("/")[-1].replace(".txt", ""), l)
                for l in data_io.read_lines(str(t_file))
            )
        }

        def get_text(f):
            key = str(f).split("/")[-1].replace(f".{audio_suffix}", "")
            return key2text[key]

        return {str(f): get_text(f) for f in audio_files}

    @staticmethod
    def get_corpora() -> List[SpeechCorpus]:
        return [TEDLIUM(n, None) for n in ["train", "dev", "test"]]

    @staticmethod
    def process_build_sample(
        audio_file, text, processed_folder, ac: AudioConfig
    ) -> ASRSample:

        si, ei = torchaudio.info(audio_file)
        num_frames = si.length / si.channels
        len_in_seconds = num_frames / si.rate
        file_name = audio_file.split("/")[-1]
        return ASRSample(file_name, text, len_in_seconds, num_frames)


CORPORA = {
    "spanish": TedxSpanish.get_corpora() + SpanishDialect.get_corpora(),
    "librispeech": LibriSpeech.get_corpora(),
    "tedlium": TEDLIUM.get_corpora(),
}

if __name__ == "__main__":

    # datasets = ["train","test" ,"dev"]
    # corpora:List[TEDLIUM] = [c for c in CORPORA["tedlium"] if c.name in datasets]
    # # corpora = CORPORA["spanish"][:1]
    #
    # dump_dir = f"{os.environ['HOME']}/data/asr_data/ENGLISH"
    # processed_folder = dump_dir
    # for c in corpora:
    #     processed_folder = f"{dump_dir}/tedlium_mp3/{c.name}"
    #     audio2text = c.build_audiofile2text(processed_folder)
    #     samples_g = (
    #     c.process_build_sample(f, t, processed_folder, AudioConfig("mp3"))._asdict() for
    #     f, t in tqdm(audio2text.items()))
    #     data_io.write_jsonl(f"{processed_folder}/{MANIFEST_FILE}", samples_g)

    datasets = ["dev-other"]
    corpora = [c for c in CORPORA["librispeech"] if c.name in datasets]
    # corpora = CORPORA["spanish"][:1]

    dump_dir = f"{os.environ['HOME']}/data/asr_data/ENGLISH"
    processed_dir = dump_dir

    audio_config = AudioConfig("mp3", 32)
    processed_dirs = [get_extract_process_zip_data(audio_config, corpus, dump_dir, processed_dir) for
     corpus in corpora]
