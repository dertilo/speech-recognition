from __future__ import annotations
import shutil

import torchaudio
from functools import partial

from tqdm import tqdm
from typing import Dict, List, NamedTuple

from abc import abstractmethod

import wget
from pathlib import Path

import os
from util import data_io
from util.util_methods import process_with_threadpool, exec_command

from data_related.utils import unzip, ASRSample, folder_to_targz, COMPRESSION_SUFFIXES
import multiprocessing

num_cpus = multiprocessing.cpu_count()

MANIFEST_FILE = "manifest.jsonl.gz"


class SpeechCorpus:
    def __init__(self, name: str, url: str) -> None:
        super().__init__()
        self.url = url
        self.name = name

    @abstractmethod
    def build_audiofile2text(self, path) -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_corpora() -> List[SpeechCorpus]:
        raise NotImplementedError


def process_write_manifest(processed_dir, file2utt, audio_conf: AudioConfig):
    os.makedirs(processed_dir, exist_ok=True)

    samples = tqdm(
        s._asdict()
        for s in process_with_threadpool(
            ({"audio_file": f, "text": t} for f, t in file2utt.items()),
            partial(
                process_build_sample, processed_folder=processed_dir, ac=audio_conf
            ),
            max_workers=2 * num_cpus,
        )
    )
    data_io.write_jsonl(f"{processed_dir}/{MANIFEST_FILE}", samples)


class AudioConfig(NamedTuple):
    format: str = "wav"
    bitrate: int = None


def process_build_sample(
    audio_file, text, processed_folder, ac: AudioConfig
) -> ASRSample:
    suffix = Path(audio_file).suffix
    assert audio_file.startswith("/")
    file_name = audio_file[1:].replace("/", "_").replace(suffix, f".{ac.format}")
    processed_audio_file = f"{processed_folder}/{file_name}"

    if ac.bitrate is not None:
        cmd = f"sox {audio_file} -C {ac.bitrate} {processed_audio_file}"
    else:
        cmd = f"sox {audio_file} {processed_audio_file}"

    exec_command(cmd)

    si, ei = torchaudio.info(processed_audio_file)
    num_frames = si.length / si.channels
    len_in_seconds = num_frames / si.rate

    return ASRSample(file_name, text, len_in_seconds, num_frames)


def maybe_download_compressed(local_filename, download_folder, url, verbose=False):
    os.makedirs(download_folder, exist_ok=True)

    suffs = [suff for suff in COMPRESSION_SUFFIXES if url.endswith(suff)]
    assert len(suffs) == 1
    suffix = suffs[0]

    localfile = os.path.join(download_folder, local_filename + suffix)

    cmd = f"wget -c -N{' -q' if not verbose else ''} -O {localfile} {url}"
    err_code = os.system(cmd)
    if err_code != 0:
        raise FileNotFoundError("could not downloaded %s" % url.split("/")[-1])
    return localfile


def prepare_corpora(
    corpora: List[SpeechCorpus],
    download_dir: str,
    processed_dir: str,
    audio_config: AudioConfig,
):
    for corpus in corpora:
        get_extract_process_zip_data(audio_config, corpus, download_dir, processed_dir)


def get_extract_process_zip_data(
    audio_config,
    corpus: SpeechCorpus,
    download_dir,
    processed_dir,
    overwrite_raw_extract=False,
):
    raw_zipfile = maybe_download_compressed(corpus.name, download_dir, corpus.url)
    ac = f"{audio_config.format}{'' if audio_config.bitrate is None else '_' + str(audio_config.bitrate)}"
    corpus_folder_name = f"{corpus.name}_processed_{ac}"
    processed_corpus_dir = os.path.join(processed_dir, corpus_folder_name)
    processed_targz = f"{download_dir}/{corpus_folder_name}.tar.gz"
    if not os.path.isfile(processed_targz):
        raw_extracted_dir = f"{processed_dir}/raw/{corpus.name}"
        if not os.path.isdir(raw_extracted_dir) or overwrite_raw_extract:
            if overwrite_raw_extract:
                shutil.rmtree(raw_extracted_dir)
            unzip(raw_zipfile, raw_extracted_dir)
        file2utt = corpus.build_audiofile2text(raw_extracted_dir)
        process_write_manifest(processed_corpus_dir, file2utt, audio_config)
        folder_to_targz(download_dir, processed_corpus_dir)
        print(f"wrote {processed_targz}")
        shutil.rmtree(raw_extracted_dir)
    else:
        print(f"found {processed_targz}")
        unzip(processed_targz, processed_dir)


def find_files_build_audio2text_openslr(
    path, parse_line_fun, audio_suffix=".wav", transcript_suffix=".tsv"
) -> Dict[str, str]:
    audio_files = [str(f) for f in Path(path).rglob(f"*{audio_suffix}")]
    assert len(audio_files) > 0

    transcript_files = list(Path(path).rglob(f"*{transcript_suffix}"))
    lines = (l for f in transcript_files for l in data_io.read_lines(str(f)))
    parsed_lines = (parse_line_fun(l) for l in lines)
    key2text = {file_name: text for file_name, text in parsed_lines}

    audio_file_key = ((f, f.split("/")[-1]) for f in audio_files)
    return {f: key2text[k] for f, k in audio_file_key}
