from __future__ import annotations
from typing import Dict, List

from datasets.common import SpeechCorpus, find_files_build_audio2text_openslr


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

if __name__ == '__main__':

    corpora = TedxSpanish.get_corpora() + SpanishDialect.get_corpora()