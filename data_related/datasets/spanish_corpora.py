from __future__ import annotations
import argparse
from typing import List, Dict
from data_related.datasets.common import SpeechCorpus, prepare_corpora, \
    find_files_build_audio2text_openslr


class SpanishDialect(SpeechCorpus):
    def build_audiofile2text(self, path) -> Dict[str, str]:
        audio_suffix = ".wav"

        def parse_line(l):
            file_name, text = l.split("\t")
            return file_name + audio_suffix, text

        return find_files_build_audio2text_openslr(path, parse_line, audio_suffix=audio_suffix)

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

        def parse_line(l):
            s = l.split(" ")
            text, file_name = s[:-1], s[-1]
            assert file_name.startswith("TEDX")
            return file_name + audio_suffix, text

        return find_files_build_audio2text_openslr(
            path,
            parse_line,
            audio_suffix=audio_suffix,
            transcript_suffix=".transcription",
        )

    @staticmethod
    def get_corpora() -> List[TedxSpanish]:
        base_url = "https://www.openslr.org/resources"
        return [TedxSpanish("67_tedx", f"{base_url}/{67}/tedx_spanish_corpus.tgz")]


parser = argparse.ArgumentParser(description="LibriSpeech Data download")
parser.add_argument("--dump_dir", required=True, default=None, type=str)
parser.add_argument("--processed_dir", required=True, default=None, type=str)
parser.add_argument("--data_sets", nargs="+", default="ALL", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    """
    python $HOME/code/SPEECH/speech-recognition/data_related/datasets/spanish_corpora.py \
    --dump_dir /tmp/SPANISH \
    --processed_dir /tmp/SPANISH \
    --data_sets "67_tedx"
    """

    dump_dir = args.dump_dir
    processed_folder = args.processed_dir

    corpora: List[SpeechCorpus] = SpanishDialect.get_corpora()
    corpora.extend(TedxSpanish.get_corpora())

    datasets = args.data_sets
    if len(datasets) > 1 or datasets[0] != "ALL":
        corpora = [c for c in corpora if c.name in datasets]

    prepare_corpora(corpora, dump_dir, processed_folder)
