from __future__ import annotations

from bs4 import BeautifulSoup
from pathlib import Path

from typing import List, Dict

import os
from util import data_io

from datasets.common import (
    SpeechCorpus,
    maybe_download,
    get_extract_process_zip_data,
    AudioConfig,
    maybe_extract,
)


class Tuda(SpeechCorpus):
    audio_suffix: str = ".wav"

    def build_audiofile2text(self, path) -> Dict[str, str]:
        audio_suffix = self.audio_suffix
        transcript_suffix = ".xml"

        audio_files = [str(f) for f in Path(path).rglob(f"*{audio_suffix}")]
        assert len(audio_files) > 0

        transcript_files = Path(path).rglob(f"*{transcript_suffix}")
        file_string = (
            (f.name, next(data_io.read_lines(str(f)))) for f in transcript_files
        )

        def parse_line_fun(f, l):
            soup = BeautifulSoup(l, "xml")
            # readme says: "sentence with the original text representation taken from the various text corpora and a cleaned version, where the sentence is normalised to resemble what speakers actually said as closely as possible. "
            text = soup.find("cleaned_sentence").text
            return f.replace(transcript_suffix, ""), text

        parsed_lines = (parse_line_fun(f, s) for f, s in file_string)
        key2text = {sampel_name: text for sampel_name, text in parsed_lines}

        def audiofile_to_key(f):
            file_name = f.split("/")[-1]
            return file_name.split("_")[0]

        audio_file_key = ((f, audiofile_to_key(f)) for f in audio_files)
        audio_files_with_trans = [
            (f, k) for f, k in audio_file_key if k in key2text.keys()
        ]
        print(f"{len(audio_files)-len(audio_files_with_trans)} have no transcripts")
        return {f: key2text[k] for f, k in audio_files_with_trans}

    @staticmethod
    def get_corpora() -> List[Tuda]:
        kaldi_tuda_de_corpus_server = (
            "http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/"
        )
        corpus_name = "german-speechdata-package-v2"
        url = f"{kaldi_tuda_de_corpus_server}/{corpus_name}.tar.gz"

        return [Tuda(ds_name, url) for ds_name in ["train", "dev", "test"]]

    def get_raw_zipfile(self, download_dir) -> str:
        local_file = f"{download_dir}/{self.url.split('/')[-1]}"
        if not os.path.isfile(local_file):
            maybe_download(local_file, self.url, False)
        return local_file

    def maybe_extract_raw(self, raw_zipfile, processed_dir):
        raw_extracted_dir = f"{processed_dir}/raw"
        maybe_extract(raw_zipfile, raw_extracted_dir)
        return f"{raw_extracted_dir}/german-speechdata-package-v2/{self.name}"


if __name__ == "__main__":
    corpora = Tuda.get_corpora()
    dump_dir = "/home/tilo/data/asr_data/GERMAN/tuda"
    # dir = "/home/tilo/data/asr_data/GERMAN/raw/german-speechdata-package-v2/dev"
    # a2t = corpus.build_audiofile2text(dir)

    audio_config = AudioConfig("wav", None)

    processed_dir = dump_dir
    for c in corpora:
        if c.name == "dev":
            c.audio_suffix = "_Samson.wav"
            get_extract_process_zip_data(
                audio_config, c, dump_dir, processed_dir, False
            )

"""
4648 have no transcripts # TODO(tilo) !?!?
18035it [04:54, 56.62it/s]formats: can't open input file `/home/tilo/data/asr_data/GERMAN/tuda/train_processed_mp3_32/home_tilo_data_asr_data_GERMAN_tuda_raw_german-speechdata-package-v2_train_2014-03-24-13-39-24_Kinect-RAW.mp3': No such file or directory
failed to process /home/tilo/data/asr_data/GERMAN/tuda/raw/german-speechdata-package-v2/train/2014-03-24-13-39-24_Kinect-RAW.wav
20780it [05:39, 61.90it/s]formats: can't open input file `/home/tilo/data/asr_data/GERMAN/tuda/train_processed_mp3_32/home_tilo_data_asr_data_GERMAN_tuda_raw_german-speechdata-package-v2_train_2014-03-27-11-50-33_Kinect-RAW.mp3': No such file or directory
failed to process /home/tilo/data/asr_data/GERMAN/tuda/raw/german-speechdata-package-v2/train/2014-03-27-11-50-33_Kinect-RAW.wav
21693it [05:54, 61.16it/s]
wrote /home/tilo/data/asr_data/GERMAN/tuda/train_processed_mp3_32.tar.gz
0it [00:00, ?it/s]0 have no transcripts
5181it [01:08, 75.36it/s]
wrote /home/tilo/data/asr_data/GERMAN/tuda/dev_processed_mp3_32.tar.gz
0 have no transcripts
5125it [01:09, 74.24it/s]
wrote /home/tilo/data/asr_data/GERMAN/tuda/test_processed_mp3_32.tar.gz
"""
