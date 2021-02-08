from __future__ import annotations

import os

from pathlib import Path

from typing import Dict, List
from util import data_io

from corpora.common import (
    SpeechCorpus,
    find_files_build_audio2text_openslr,
    AudioConfig,
    get_extract_process_zip_data,
)


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


class CommonVoiceSpanish(SpeechCorpus):
    @staticmethod
    def common_voice_data(path, split_name: str):
        g = data_io.read_lines(
            os.path.join(path, f"{path}/cv-corpus-6.1-2020-12-11/es/{split_name}.tsv")
        )
        header = next(g).split("\t")

        def parse_line(l):
            d = {k: v for k, v in zip(header, l.split("\t"))}
            return d

        return map(parse_line, g)

    def build_audiofile2text(self, path) -> Dict[str, str]:
        key2utt = {
            d["path"]: d["sentence"] for d in self.common_voice_data(path, self.name)
        }
        utts = list(Path(path).rglob("*.mp3"))

        def get_key(f):
            return str(f).split("/")[-1]

        malpaudios = [
            "common_voice_es_19499901.mp3",
            "common_voice_es_19499893.mp3",
        ]  # broken audios

        return {
            str(f): key2utt[get_key(f)]
            for f in utts
            if get_key(f) in key2utt.keys() and str(f).split("/")[-1] not in malpaudios
        }

    def get_raw_zipfile(self, download_dir) -> str:
        """
        manually download es.tar.gz from: https://voice.mozilla.org/en/datasets
        cause mozilla wants your email before they let you download the corpus!
        """
        return f"{download_dir}/es.tar.gz"

    @staticmethod
    def get_corpora() -> List[CommonVoiceSpanish]:
        return [CommonVoiceSpanish(ds) for ds in ["train", "dev", "test"]]


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


if __name__ == "__main__":

    # corpora = TedxSpanish.get_corpora() + SpanishDialect.get_corpora()

    corpora = CommonVoiceSpanish.get_corpora()
    audio_config = AudioConfig("mp3")
    for corpus in corpora:
        get_extract_process_zip_data(
            audio_config,
            corpus,
            f"/data",
            f"/data/SPANISH_CV",
            remove_raw_extract=False,
            overwrite=True,
        )
