import os

import tarfile

from zipfile import ZipFile

from typing import List, Dict, Callable, Iterable, NamedTuple


class Sample(NamedTuple):
    audio_file: str
    text: str
    length: float  # in seconds
    num_frames:int


def unzip(zipfile: str, dest_dir: str) -> None:

    if any([zipfile.endswith(s) for s in [".zip", ".ZIP"]]):
        with ZipFile(zipfile, "r") as zipObj:
            zipObj.extractall(dest_dir)
    elif any([zipfile.endswith(s) for s in [".tar.gz", ".TAR.GZ",".tgz"]]):
        with tarfile.open(zipfile, mode="r:gz") as tar:
            tar.extractall(dest_dir)


def folder_to_targz(destination_path, source_dir):
    folder_name = os.path.basename(source_dir)
    with tarfile.open(f"{destination_path}/{folder_name}.tar.gz", "w:gz") as tar:
        tar.add(source_dir, arcname=folder_name)
