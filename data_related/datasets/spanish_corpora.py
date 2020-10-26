import argparse

from functools import partial

import shutil

import torchaudio
from tqdm import tqdm
from typing import List, Dict, Tuple

from pathlib import Path

import os
from util import data_io
from util.util_methods import process_with_threadpool, exec_command

from data_related.datasets.common import SpeechCorpus
from data_related.utils import Sample, unzip, folder_to_targz

class SpanishDialect(SpeechCorpus):

    def build_audiofile2text(self,path)->Dict[str,str]:
        return read_openslr(path)

class TedxSpanish(SpeechCorpus):


    def __init__(self) -> None:
        base_url = "https://www.openslr.org/resources"
        super().__init__("67_tedx", f"{base_url}/{67}/tedx_spanish_corpus.tgz")

    def build_audiofile2text(self, path) -> Dict[str, str]:
        return read_openslr(path,transcript_suffix=".transcription")


def build_spanish_latino_speech_corpora()->List[SpeechCorpus]:
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

    return [SpeechCorpus(n,u) for n,u in name_urls.items()]


def read_openslr(path,audio_suffix=".wav",transcript_suffix=".tsv") -> Dict[str, str]:
    wavs = list(Path(path).rglob(audio_suffix))
    tsvs = list(Path(path).rglob(transcript_suffix))

    def parse_line(l):
        file_name, text = l.split("\t")
        return file_name + ".wav", text

    key2text = {
        file_name: text
        for tsv_file in tsvs
        for file_name, text in (
            parse_line(l) for l in data_io.read_lines(os.path.join(path, str(tsv_file)))
        )
    }

    def get_text(f):
        key = str(f).split("/")[-1]
        return key2text[key]

    return {str(f): get_text(f) for f in wavs}



parser = argparse.ArgumentParser(description="LibriSpeech Data download")
parser.add_argument("--dump_dir", required=True, default=None, type=str)
parser.add_argument("--processed_dir", required=True, default=None, type=str)
parser.add_argument("--data_sets", nargs="+", default="ALL", type=str)
args = parser.parse_args()

if __name__ == "__main__":

    dump_dir = args.dump_dir
    os.makedirs(dump_dir, exist_ok=True)

    processed_folder = args.processed_dir
    os.makedirs(processed_folder, exist_ok=True)

    corpora:List[SpeechCorpus] = build_spanish_latino_speech_corpora()
    corpora.append(TedxSpanish())

    datasets = args.data_sets
    if len(datasets) > 1 or datasets[0] != "ALL":
        corpora = [c for c in corpora if c.name in datasets]


    for corpus in corpora:
        raw_zipfile = corpus.maybe_download(dump_dir)

        extract_folder = f"/{processed_folder}/raw/{corpus.name}"
        corpus_folder = os.path.join(processed_folder, f"{corpus.name}_processed")
        os.makedirs(corpus_folder, exist_ok=True)
        dumped_targz_file = f"{dump_dir}/{corpus.name}_processed.tar.gz"
        if not os.path.isfile(dumped_targz_file):
            corpus.extract_downloaded(raw_zipfile, extract_folder)
            file2utt = corpus.build_audiofile2text(extract_folder)
            corpus.process_write_manifest(corpus_folder, file2utt)
            folder_to_targz(dump_dir, corpus_folder)
            print(f"wrote {dumped_targz_file}")
            shutil.rmtree(extract_folder)
        else:
            print(f"found {dumped_targz_file}")
            unzip(dumped_targz_file, processed_folder)

