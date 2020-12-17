# stolen from nvidia nemo
import argparse
import fnmatch
import json
import os
import subprocess
import tarfile
import urllib.request

from sox import Transformer
from tqdm import tqdm


URLS = {
    "TRAIN_CLEAN_100": ("http://www.openslr.org/resources/12/train-clean-100.tar.gz"),
    "TRAIN_CLEAN_360": ("http://www.openslr.org/resources/12/train-clean-360.tar.gz"),
    "TRAIN_OTHER_500": ("http://www.openslr.org/resources/12/train-other-500.tar.gz"),
    "DEV_CLEAN": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "DEV_OTHER": "http://www.openslr.org/resources/12/dev-other.tar.gz",
    "TEST_CLEAN": "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "TEST_OTHER": "http://www.openslr.org/resources/12/test-other.tar.gz",
}


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    source = URLS[source]
    if not os.path.exists(destination):
        urllib.request.urlretrieve(source, filename=destination + ".tmp")
        os.rename(destination + ".tmp", destination)
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        pass


def __process_data(
    data_folder: str, dst_folder: str, manifest_file: str, audio_format="wav", bits=None
):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
    Returns:
    """
    bitss = "" if bits is None else f"_{bits}"
    dst_folder = f"{dst_folder}_{audio_format}{bitss}"
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []
    entries = []

    for root, dirnames, filenames in os.walk(data_folder):
        for filename in fnmatch.filter(filenames, "*.trans.txt"):
            files.append((os.path.join(root, filename), root))

    for transcripts_file, root in tqdm(files):
        with open(transcripts_file, encoding="utf-8") as fin:
            for line in fin:
                id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
                transcript_text = text.lower().strip()

                # Convert FLAC file to WAV
                flac_file = os.path.join(root, id + ".flac")
                wav_file = os.path.join(dst_folder, f"{id}.{audio_format}")
                if not os.path.exists(wav_file):
                    if bits is not None:
                        cmd = f"sox {flac_file} -C {bits} {wav_file}"
                    else:
                        cmd = f"sox {flac_file} {wav_file}"
                    assert os.system(cmd) == 0
                    # audio_converter.build(flac_file, wav_file,extra_args=["-C",128])
                # check duration
                duration = subprocess.check_output(
                    "soxi -D {0}".format(wav_file), shell=True
                )

                entry = {}
                entry["audio_filepath"] = os.path.abspath(wav_file)
                entry["duration"] = float(duration)
                entry["text"] = transcript_text
                entries.append(entry)

    with open(f"{manifest_file}", "w") as fout:
        for m in entries:
            fout.write(json.dumps(m) + "\n")


def main(
    data_root="/tmp/asr_data/ENGLISH",
    data_sets="dev_other",
    bits=32,
    format="wav",
):
    os.makedirs(data_root, exist_ok=True)
    bitss = f"_{bits}" if bits is not None else ""

    if data_sets == "ALL":
        data_sets = "dev_clean,dev_other,train_clean_100,train_clean_360,train_other_500,test_clean,test_other"

    for data_set in data_sets.split(","):
        filepath = os.path.join(data_root, data_set + ".tar.gz")
        __maybe_download_file(filepath, data_set.upper())
        __extract_file(filepath, data_root)
        data_folder = os.path.join(
            os.path.join(data_root, "LibriSpeech"),
            data_set.replace("_", "-"),
        )
        __process_data(
            data_folder,
            data_folder + "-processed",
            os.path.join(data_root, data_set + f"_{format}{bitss}.json"),
            audio_format=format,
            bits=bits,
        )
        # shutil.rmtree(data_folder)


if __name__ == "__main__":
    main()
