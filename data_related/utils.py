from dataclasses import dataclass

import os

import tarfile

from zipfile import ZipFile

from typing import List, Dict, Callable, Iterable, NamedTuple


@dataclass(frozen=True, eq=True)
class ASRSample:
    audio_file: str
    text: str
    duration: float  # in seconds
    num_frames:int

ZIP_SUFFIXES = [".zip", ".ZIP"]
TAR_GZ_SUFFIXES = [".tar.gz", ".TAR.GZ",".tgz"]
COMPRESSION_SUFFIXES = ZIP_SUFFIXES + TAR_GZ_SUFFIXES

def unzip(zipfile: str, dest_dir: str) -> None:

    os.makedirs(dest_dir, exist_ok=True)

    if any([zipfile.endswith(s) for s in ZIP_SUFFIXES ]):
        with ZipFile(zipfile, "r") as zipObj:
            zipObj.extractall(dest_dir)
    elif any([zipfile.endswith(s) for s in TAR_GZ_SUFFIXES]):
        with tarfile.open(zipfile, mode="r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, dest_dir)
    else:
        raise NotImplementedError


def folder_to_targz(source_dir, destination_path):
    folder_name = os.path.basename(source_dir)
    with tarfile.open(f"{destination_path}/{folder_name}.tar.gz", "w:gz") as tar:
        tar.add(source_dir, arcname=folder_name)
