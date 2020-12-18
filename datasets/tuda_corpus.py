from __future__ import annotations

from bs4 import BeautifulSoup
from pathlib import Path

from typing import List, Dict

import os
from util import data_io

from datasets.common import maybe_download_compressed, SpeechCorpus


class Tuda(SpeechCorpus):
    def build_audiofile2text(self, path) -> Dict[str, str]:
        audio_suffix = ".wav"
        transcript_suffix = ".xml"

        audio_files = [str(f) for f in Path(path).rglob(f"*{audio_suffix}")]
        assert len(audio_files) > 0

        transcript_files = Path(path).rglob(f"*{transcript_suffix}")
        file_string = (
            (f.name, next(data_io.read_lines(str(f)))) for f in transcript_files
        )

        def parse_line_fun(f, l):
            soup = BeautifulSoup(l, "xml")
            # readme says: "sentence with the original text representation taken from the various text corpora and a cleaned version, where the sentence is normalised to resemble what speakers actually said as closely as possible. "
            text = soup.find("cleaned_sentence").text
            return f.replace(transcript_suffix, ""), text

        parsed_lines = (parse_line_fun(f, s) for f, s in file_string)
        key2text = {sampel_name: text for sampel_name, text in parsed_lines}

        def audiofile_to_key(f):
            file_name = f.split("/")[-1]
            return file_name.split("_")[0]

        audio_file_key = ((f, audiofile_to_key(f)) for f in audio_files)
        return {f: key2text[k] for f, k in audio_file_key}

    @staticmethod
    def get_corpora() -> List[Tuda]:
        kaldi_tuda_de_corpus_server = (
            "http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/"
        )
        dataset_name = "german-speechdata-package-v2"
        url = f"{kaldi_tuda_de_corpus_server}/{dataset_name}.tar.gz"

        return [Tuda(dataset_name, url)]


if __name__ == "__main__":
    corpus = Tuda.get_corpora()[0]
    dir = "/home/tilo/data/asr_data/GERMAN/raw/german-speechdata-package-v2/dev"
    a2t = corpus.build_audiofile2text(dir)
