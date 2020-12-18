from __future__ import annotations

import torchaudio
from pathlib import Path

import os

import argparse
from tqdm import tqdm
from typing import List, Dict
from util import data_io

from datasets.common import (
    SpeechCorpus,
    prepare_corpora,
    find_files_build_audio2text_openslr,
    AudioConfig,
    MANIFEST_FILE,
)
from data_related.utils import ASRSample


class SpanishDialect(SpeechCorpus):
    def build_audiofile2text(self, path) -> Dict[str, str]:
        audio_suffix = ".wav"
        transcript_suffix = ".tsv"

        def parse_line(l):
            file_name, text = l.split("\t")
            return file_name + audio_suffix, text

        return find_files_build_audio2text_openslr(
            path,
            parse_line,
            audio_suffix=audio_suffix,
            transcript_suffix=transcript_suffix,
        )

    @staticmethod
    def get_corpora() -> List[SpanishDialect]:
        base_url = "https://www.openslr.org/resources"
        name_urls = {
            f"{eid}_{abbrev}_{sex}": f"{base_url}/{eid}/es_{abbrev}_{sex}.zip"
            for eid, abbrev in [
                ("71", "cl"),  # chilean
                ("72", "co"),  # colombian
                ("73", "pe"),  # peruvian
                ("74", "pr"),  # puerto rico
                ("75", "ve"),  # venezuelan
                ("61", "ar"),  # argentinian
            ]
            for sex in ["male", "female"]
            if not (eid == "74" and sex == "male")  # cause 74 has no male speaker
        }
        return [SpanishDialect(n, u) for n, u in name_urls.items()]


class TedxSpanish(SpeechCorpus):
    def build_audiofile2text(self, path) -> Dict[str, str]:
        audio_suffix = ".wav"
        transcript_suffix = ".transcription"

        def parse_line(l):
            s = l.split(" ")
            text, file_name = s[:-1], s[-1]
            assert file_name.startswith("TEDX")
            return file_name + audio_suffix, text

        return find_files_build_audio2text_openslr(
            path,
            parse_line,
            audio_suffix=audio_suffix,
            transcript_suffix=transcript_suffix,
        )

    @staticmethod
    def get_corpora() -> List[TedxSpanish]:
        base_url = "https://www.openslr.org/resources"
        return [TedxSpanish("67_tedx", f"{base_url}/{67}/tedx_spanish_corpus.tgz")]


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


class Caito(SpeechCorpus):
    def build_audiofile2text(self, path) -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    def get_corpora() -> List[SpeechCorpus]:
        url = "http://www.caito.de/data/Training/stt_tts"
        return [
            Caito(n, f"{url}/{n}.tgz") for n in ["es_ES", "en_US", "en_UK", "de_DE"]
        ]


class HeroicoUSMA(SpeechCorpus):
    def build_audiofile2text(self, path) -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    def get_corpora() -> List[SpeechCorpus]:
        url = "http://www.openslr.org/resources/39"
        return [HeroicoUSMA("heroico", f"{url}/LDC2006S37.tar.gz")]


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
    processed_folder = dump_dir

    prepare_corpora(corpora, dump_dir, processed_folder, AudioConfig("mp3", 32))
