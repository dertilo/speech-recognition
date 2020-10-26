import wget
from pathlib import Path

import os


def maybe_download(dataset_url, download_folder):
    corpusname_file = []
    for data_set, url in dataset_url:
        localfile = os.path.join(download_folder, data_set + Path(url).suffix)
        if not os.path.exists(localfile):
            print(f"downloading: {url}")
            wget.download(url, localfile)
        else:
            print(f"found: {localfile} no need to download")
        corpusname_file.append((data_set, localfile))
    return corpusname_file


def build_dataset_url(datasets, name_urls):
    if len(datasets) == 1 and datasets[0] == "ALL":
        datasets = list(name_urls.keys())
    else:
        assert all([k in name_urls.keys() for k in datasets])
    dataset_url = [(ds, name_urls[ds]) for ds in datasets]
    return dataset_url


def download_data(datasets, download_folder, name_urls):
    os.makedirs(download_folder, exist_ok=True)
    dataset_url = build_dataset_url(datasets, name_urls)
    corpusname_file = maybe_download(dataset_url, download_folder)
    return corpusname_file
