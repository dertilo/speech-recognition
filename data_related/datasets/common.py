from __future__ import annotations
import shutil

import torchaudio
from functools import partial

from tqdm import tqdm
from typing import Dict, List

from abc import abstractmethod

import wget
from pathlib import Path

import os
from util import data_io
from util.util_methods import process_with_threadpool, exec_command

from data_related.utils import unzip, Sample, folder_to_targz, COMPRESSION_SUFFIXES
import multiprocessing

num_cpus = multiprocessing.cpu_count()

MANIFEST_FILE = "manifest.jsonl.gz"

class SpeechCorpus:

    def __init__(self, name: str, url: str) -> None:
        super().__init__()
        self.url = url
        self.name = name
        suffs = [suff for suff in COMPRESSION_SUFFIXES if self.url.endswith(suff)]
        assert len(suffs)==1
        self.suffix = suffs[0]

    def maybe_download(self, download_folder)->str:
        return maybe_download(self.name, download_folder, self.url,self.suffix)

    @staticmethod
    def extract_downloaded(raw_zipfile,extract_folder):
        unzip(raw_zipfile, extract_folder)

    @staticmethod
    def process_write_manifest(corpus_folder, file2utt):
        samples = tqdm(
            s._asdict()
            for s in process_with_threadpool(
                ({"audio_file": f, "text": t} for f, t in file2utt.items()),
                partial(convert_to_mp3_get_length, processed_folder=corpus_folder),
                max_workers=2 * num_cpus,
            )
        )
        data_io.write_jsonl(f"{corpus_folder}/{MANIFEST_FILE}", samples)

    @abstractmethod
    def build_audiofile2text(self, path) -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_corpora()->List[SpeechCorpus]:
        raise NotImplementedError

def convert_to_mp3_get_length(audio_file, text, processed_folder) -> Sample:
    suffix = Path(audio_file).suffix
    assert audio_file.startswith("/")
    mp3_file_name = audio_file[1:].replace("/", "_").replace(suffix, ".mp3")

    mp3_file = f"{processed_folder}/{mp3_file_name}"
    exec_command(f"sox {audio_file} {mp3_file}")

    si, ei = torchaudio.info(mp3_file)
    num_frames = si.length / si.channels
    len_in_seconds = num_frames / si.rate

    return Sample(mp3_file_name, text, len_in_seconds, num_frames)

def maybe_download(data_set, download_folder, url,suffix):
    localfile = os.path.join(download_folder, data_set + suffix)
    if not os.path.exists(localfile):
        print(f"downloading: {url}")
        wget.download(url, localfile)
    else:
        print(f"found: {localfile} no need to download")
    return localfile


def prepare_corpora(corpora:List[SpeechCorpus],dump_dir:str,processed_folder:str):
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    for corpus in corpora:
        raw_zipfile = corpus.maybe_download(dump_dir)

        extract_folder = f"{processed_folder}/raw/{corpus.name}"
        os.makedirs(extract_folder, exist_ok=True)
        corpus_folder = os.path.join(processed_folder, f"{corpus.name}_processed")
        os.makedirs(corpus_folder, exist_ok=True)
        dumped_targz_file = f"{dump_dir}/{corpus.name}_processed.tar.gz"
        if not os.path.isfile(dumped_targz_file):
            #corpus.extract_downloaded(raw_zipfile, extract_folder)
            file2utt = corpus.build_audiofile2text(extract_folder)
            corpus.process_write_manifest(corpus_folder, file2utt)
            folder_to_targz(dump_dir, corpus_folder)
            print(f"wrote {dumped_targz_file}")
            # shutil.rmtree(extract_folder)
        else:
            print(f"found {dumped_targz_file}")
            unzip(dumped_targz_file, processed_folder)


def find_files_build_audio2text_openslr(
    path, parse_line_fun, audio_suffix=".wav", transcript_suffix=".tsv"
) -> Dict[str, str]:
    def build_file2text(parse_line, transcripts, audios):
        key2text = {
            file_name: text
            for tsv_file in transcripts
            for file_name, text in (
                parse_line(l) for l in data_io.read_lines(str(tsv_file))
            )
        }

        def get_text(f):
            key = str(f).split("/")[-1]
            return key2text[key]

        return {str(f): get_text(f) for f in audios}
    # ------------------------------------------------------------------------
    audio_files = list(Path(path).rglob(f"*{audio_suffix}"))
    assert len(audio_files)
    transcript_files = list(Path(path).rglob(f"*{transcript_suffix}"))
    return build_file2text(parse_line_fun, transcript_files, audio_files)