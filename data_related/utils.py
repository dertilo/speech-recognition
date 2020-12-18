import os

import tarfile

from zipfile import ZipFile

from typing import List, Dict, Callable, Iterable, NamedTuple


class ASRSample(NamedTuple):
    audio_file: str
    text: str
    duration: float  # in seconds
    num_frames:int

ZIP_SUFFIXES = [".zip", ".ZIP"]
TAR_GZ_SUFFIXES = [".tar.gz", ".TAR.GZ",".tgz"]
COMPRESSION_SUFFIXES = ZIP_SUFFIXES + TAR_GZ_SUFFIXES

def unzip(zipfile: str, dest_dir: str) -> None:

    if any([zipfile.endswith(s) for s in ZIP_SUFFIXES ]):
        with ZipFile(zipfile, "r") as zipObj:
            zipObj.extractall(dest_dir)
    elif any([zipfile.endswith(s) for s in TAR_GZ_SUFFIXES]):
        with tarfile.open(zipfile, mode="r:gz") as tar:
            tar.extractall(dest_dir)
    else:
        raise NotImplementedError


def folder_to_targz(destination_path, source_dir):
    folder_name = os.path.basename(source_dir)
    with tarfile.open(f"{destination_path}/{folder_name}.tar.gz", "w:gz") as tar:
        tar.add(source_dir, arcname=folder_name)
