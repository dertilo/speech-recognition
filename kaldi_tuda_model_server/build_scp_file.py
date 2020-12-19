import os

from pathlib import Path
from tqdm import tqdm
from util import data_io

from data_related.utils import ASRSample


def build_line(f):
    file = str(f).replace(os.environ["HOME"], "/docker-share")
    return f"{f.name.replace('.wav','')} {file}"


def build_scp_from_original_files(base_path):
    dataset_name = "test"
    tech = "Yamaha"
    p = f"{base_path}/{dataset_name}"
    data_io.write_lines(
        f"{base_path}/wav_{dataset_name}_{tech}.scp",
        (build_line(f) for f in Path(p).rglob(f"*_{tech}.wav")),
    )


if __name__ == "__main__":
    base_path = f"{os.environ['HOME']}/data/asr_data/GERMAN/tuda"
    # base_path = (
    #     f"{os.environ['HOME']}/data/asr_data/GERMAN/tuda/raw/german-speechdata-package-v2"
    # )
    # build_scp_from_original_files(base_path)
    folder = "dev_processed_wav"
    asr_samples = (
        ASRSample(**s)
        for s in data_io.read_jsonl(f"{base_path}/{folder}/manifest.jsonl.gz")
    )
    data_io.write_lines(
        f"{base_path}/{folder}.scp",
        tqdm(
            f"{sample.audio_file} /docker-share/data/asr_data/GERMAN/tuda/{folder}/{sample.audio_file}"
            for sample in asr_samples
            if "Samson" in sample.audio_file
        ),
    )
