from __future__ import print_function

import fnmatch
import gzip
import io
import json
import os

from scipy.io.wavfile import read
from tqdm import tqdm
import subprocess
import torch.distributed as dist
from typing import Dict, Iterable


def write_lines(file, lines: Iterable[str], mode="wb"):
    def process_line(line):
        line = line + "\n"
        return line.encode("utf-8")

    with gzip.open(file, mode=mode) if file.endswith(".gz") else open(
        file, mode=mode
    ) as f:
        f.writelines((process_line(l) for l in lines))


def read_jsonl(file, mode="b", limit=None, num_to_skip=0):
    assert any([mode == m for m in ["b", "t"]])
    with gzip.open(file, mode="r" + mode) if file.endswith(".gz") else open(
        file, mode="rb"
    ) as f:
        [next(f) for _ in range(num_to_skip)]
        for k, line in enumerate(f):
            if limit and (k >= limit):
                break
            yield json.loads(line.decode("utf-8") if mode == "b" else line)


def write_json(file: str, datum: Dict, mode="wb"):
    with gzip.open(file, mode=mode) if file.endswith("gz") else open(
        file, mode=mode
    ) as f:
        line = json.dumps(datum, skipkeys=True, ensure_ascii=False)
        if "b" in mode:
            line = line.encode("utf-8")
        f.write(line)


def create_manifest(data_path, output_path, min_duration=None, max_duration=None):
    file_paths = [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(data_path)
        for f in fnmatch.filter(files, "*.wav")
    ]
    file_paths = order_and_prune_files(file_paths, min_duration, max_duration)
    with io.FileIO(output_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace("/wav/", "/txt/").replace(".wav", ".txt")
            sample = (
                os.path.abspath(wav_path)
                + ","
                + os.path.abspath(transcript_path)
                + "\n"
            )
            file.write(sample.encode("utf-8"))
    print("\n")


def order_and_prune_files(file_paths, min_duration, max_duration):
    print("Sorting manifests...")
    duration_file_paths = [
        (
            path,
            float(subprocess.check_output(['soxi -D "%s"' % path.strip()], shell=True)),
        )
        for path in tqdm(file_paths)
    ]
    if min_duration and max_duration:
        print(
            "Pruning manifests between %d and %d seconds" % (min_duration, max_duration)
        )
        duration_file_paths = [
            (path, duration)
            for path, duration in duration_file_paths
            if min_duration <= duration <= max_duration
        ]

    def func(element):
        return element[1]

    duration_file_paths.sort(key=func)
    return [x[0] for x in duration_file_paths]  # Remove durations


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def load_audio(path):
    sample_rate, sound = read(path)
    sound = sound.astype("float32") / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound