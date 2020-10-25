from typing import Union, List

import wget
from pathlib import Path

import os


def download_spanish_srl_corpora(
    datasets: Union[str, List[str]] = "ALL", download_folder="/tmp"
):
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
    name_urls["67_tedx"] = "tedx_spanish_corpus.tgz"
    assert all([k in name_urls.keys() for k in datasets])

    datasets = list(name_urls.keys()) if datasets == "ALL" else datasets
    for data_set in datasets:
        url = name_urls[data_set]
        localfile = os.path.join(download_folder, data_set + Path(url).suffix)
        if not os.path.exists(localfile):
            wget.download(url, localfile)


if __name__ == "__main__":
    download_spanish_srl_corpora(["74_pr_female"], "/tmp")
