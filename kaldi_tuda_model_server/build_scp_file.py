import os

from pathlib import Path
from util import data_io


def build_line(f):
    file = str(f).replace(os.environ["HOME"], "/docker-share")
    return f"{f.name.replace('.wav','')} {file}"


if __name__ == "__main__":
    base_path = (
        f"{os.environ['HOME']}/data/asr_data/GERMAN/german-speechdata-package-v2"
    )
    dataset_name = "test"
    tech = "Yamaha"
    p = f"{base_path}/{dataset_name}"
    data_io.write_lines(
        f"{base_path}/wav_{dataset_name}_{tech}.scp",
        (
            build_line(f)
            for f in Path(p).rglob(f"*_{tech}.wav")
        ),
    )
