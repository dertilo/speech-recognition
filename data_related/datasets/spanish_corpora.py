import argparse

from functools import partial

import shutil

import torchaudio
from tqdm import tqdm
from typing import Union, List, Dict, Generator, Tuple

import wget
from pathlib import Path

import os
from util import data_io
from util.util_methods import process_with_threadpool, exec_command

from data_related.utils import Sample, unzip, folder_to_targz


def download_spanish_srl_corpora(
    datasets: List[str] = ["ALL"], download_folder="/tmp"
):
    os.makedirs(download_folder, exist_ok=True)

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
    name_urls["67_tedx"] = f"{base_url}/{67}/tedx_spanish_corpus.tgz"

    if len(datasets) == 1 and datasets[0] == "ALL":
        datasets = list(name_urls.keys())
    else:
        assert all([k in name_urls.keys() for k in datasets])

    corpusname_file = []
    for data_set in datasets:
        url = name_urls[data_set]
        localfile = os.path.join(download_folder, data_set + Path(url).suffix)
        if not os.path.exists(localfile):
            print(f"downloading: {url}")
            wget.download(url, localfile)
        else:
            print(f"found: {localfile} no need to download")
        corpusname_file.append((data_set, localfile))
    return corpusname_file


def read_openslr(path) -> Dict[str, str]:
    wavs = list(Path(path).rglob("*.wav"))
    tsvs = list(Path(path).rglob("*.tsv"))

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


MANIFEST_FILE = "manifest.jsonl.gz"


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


import multiprocessing

num_cpus = multiprocessing.cpu_count()


def process_data(
    corpusname_file: List[Tuple[str, str]], processed_folder, targz_dump_dir=None
):

    for corpusname, raw_zipfile in corpusname_file:
        extract_folder = f"/{processed_folder}/raw/{corpusname}"
        corpus_folder = os.path.join(processed_folder, f"{corpusname}_processed")
        unzip(raw_zipfile, extract_folder)
        file2utt = read_openslr(extract_folder)

        process_write_manifest(corpus_folder, file2utt)
        if targz_dump_dir is not None:
            folder_to_targz(targz_dump_dir, corpus_folder)
        shutil.rmtree(extract_folder)


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


parser = argparse.ArgumentParser(description="LibriSpeech Data download")
parser.add_argument("--dump_dir", required=True, default=None, type=str)
parser.add_argument("--processed_dir", required=True, default=None, type=str)
parser.add_argument("--data_sets", nargs='+', default="ALL", type=str)
args = parser.parse_args()

if __name__ == "__main__":

    corpusname_file = download_spanish_srl_corpora(args.data_sets, args.dump_dir)
    os.makedirs(args.processed_dir, exist_ok=True)
    process_data(corpusname_file,args.processed_dir,targz_dump_dir=args.dump_dir)
