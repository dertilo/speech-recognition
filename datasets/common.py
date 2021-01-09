from __future__ import annotations

from dataclasses import dataclass, asdict

import shutil

import torchaudio
from functools import partial

from tqdm import tqdm
from typing import Dict, List, NamedTuple, Tuple, Optional

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


@dataclass
class SpeechCorpus:
    name: str
    url: str

    @abstractmethod
    def build_audiofile2text(self, path) -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_corpora() -> List[SpeechCorpus]:
        raise NotImplementedError

    def get_raw_zipfile(self, download_dir) -> str:
        return maybe_download_compressed(self.name, download_dir, self.url)

    def maybe_extract_raw(self, raw_zipfile, processed_dir):
        raw_extracted_dir = f"{processed_dir}/raw/{self.name}"
        maybe_extract(raw_zipfile, raw_extracted_dir, False)
        return raw_extracted_dir


def process_write_manifest(
    raw_processed_dir: Tuple[str, str], file2utt, audio_conf: AudioConfig
):
    raw_dir, processed_dir = raw_processed_dir
    os.makedirs(processed_dir, exist_ok=True)
    failed = lambda x: x is None
    samples = tqdm(
        asdict(s)
        for s in process_with_threadpool(
            ({"audio_file": f, "text": t} for f, t in file2utt.items()),
            partial(
                process_build_sample,
                raw_processed_dir=raw_processed_dir,
                ac=audio_conf,
            ),
            max_workers=2 * num_cpus,
        )
        if not failed(s)
    )
    data_io.write_jsonl(f"{processed_dir}/{MANIFEST_FILE}", samples)



@dataclass(frozen=True, eq=True)
class AudioConfig:
    format: str = "wav"
    bitrate: Optional[int] = None
    min_dur_secs:float = 0.5 # seconds


def process_build_sample(
    audio_file, text, raw_processed_dir: Tuple, ac: AudioConfig
) -> ASRSample:
    try:
        file_name, len_in_seconds, num_frames = process_audio(
            audio_file, raw_processed_dir, ac
        )
        assert len_in_seconds>ac.min_dur_secs
        asr_sample = ASRSample(file_name, text, len_in_seconds, num_frames)
    except Exception:
        print(f"failed to process {audio_file}")
        asr_sample = None
    return asr_sample


def process_audio(audio_file, raw_processed_dir: Tuple, ac: AudioConfig):
    raw_dir, processed_dir = raw_processed_dir
    assert audio_file.startswith(raw_dir)
    suffix = Path(audio_file).suffix
    file_name = (
        audio_file.replace(f"{raw_dir}/", "")
        .replace("/", "_")
        .replace(suffix, f".{ac.format}")
    )
    processed_audio_file = f"{processed_dir}/{file_name}"
    if ac.bitrate is not None:
        cmd = f"sox {audio_file} -C {ac.bitrate} {processed_audio_file}"
    else:
        cmd = f"sox {audio_file} {processed_audio_file}"
    exec_command(cmd)
    si, ei = torchaudio.info(processed_audio_file)
    num_frames = int(si.length / si.channels)
    len_in_seconds = num_frames / si.rate
    return file_name, len_in_seconds, num_frames


def maybe_download_compressed(local_filename, download_folder, url, verbose=False):
    os.makedirs(download_folder, exist_ok=True)

    suffs = [suff for suff in COMPRESSION_SUFFIXES if url.endswith(suff)]
    assert len(suffs) == 1
    suffix = suffs[0]

    localfile = os.path.join(download_folder, local_filename + suffix)

    maybe_download(localfile, url, verbose)
    return localfile


def maybe_download(localfile, url, verbose):
    cmd = f"wget -c -N{' -q' if not verbose else ''} -O {localfile} {url}"
    err_code = os.system(cmd)
    if err_code != 0:
        raise FileNotFoundError("could not downloaded %s" % url.split("/")[-1])


def get_extract_process_zip_data(
    audio_config: AudioConfig,
    corpus: SpeechCorpus,
    dump_dir: str,
    work_dir: str,
    remove_raw_extract=True,
    overwrite:bool=False
):
    """
    dump_dir: only zipped archives files here, NO unzipping/extracting! -> used for google-drive
    work_dir: extracting+processing here, but volatile! -> content-dir on colab compute machine
    """
    raw_zipfile = corpus.get_raw_zipfile(dump_dir)
    ac = f"{audio_config.format}{'' if audio_config.bitrate is None else '_' + str(audio_config.bitrate)}"
    corpus_folder_name = f"{corpus.name}_processed_{ac}"
    processed_targz = f"{dump_dir}/{corpus_folder_name}.tar.gz"
    if not os.path.isfile(processed_targz) or overwrite:
        processed_corpus_dir = os.path.join(work_dir, corpus_folder_name)
        raw_data_dir = corpus.maybe_extract_raw(raw_zipfile, work_dir)
        file2utt = corpus.build_audiofile2text(raw_data_dir)
        process_write_manifest(
            (raw_data_dir, processed_corpus_dir), file2utt, audio_config
        )
        print(f"targzipping {processed_corpus_dir}")
        folder_to_targz(processed_corpus_dir, dump_dir)
        print(f"wrote {processed_targz}")
        if remove_raw_extract:
            shutil.rmtree(raw_data_dir)
    else:
        print(f"found {processed_targz}")
        unzip(processed_targz, work_dir)


def maybe_extract(raw_zipfile, raw_extracted_dir, overwrite_raw_extract=False):
    if not os.path.isdir(raw_extracted_dir) or overwrite_raw_extract:
        if overwrite_raw_extract:
            shutil.rmtree(raw_extracted_dir)
        unzip(raw_zipfile, raw_extracted_dir)


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
