from __future__ import absolute_import, division, print_function, unicode_literals

import os
from time import time
from typing import List
from tqdm import tqdm
from util import data_io
from util.util_methods import process_with_threadpool
from corpora.librispeech import librispeech_corpus
from data_related.audio_feature_extraction import get_length
from data_related.audio_util import Sample
from utils import HOME, BLANK_SYMBOL, SPACE


def load_samples(file: str, base_path: str) -> List[Sample]:
    def adjust_file_path(d):
        d["audio_file"] = os.path.join(base_path, d["audio_file"])
        return Sample(**d)

    return [adjust_file_path(d) for d in data_io.read_jsonl(file)]


def build_librispeech_corpus(
    raw_data_path, name: str, folders: List[str], reprocess=False,
) -> List[Sample]:
    file = raw_data_path + "/%s_samples.jsonl.gz" % name

    if os.path.isfile(file) and not reprocess:
        print("loading processed samples from %s" % file)
        samples = load_samples(file, raw_data_path)
    else:
        corpus = {
            k: v
            for folder in folders
            for k, v in librispeech_corpus(os.path.join(raw_data_path, folder)).items()
        }

        assert len(corpus) > 0

        def build_sample(audio_file, text):
            return Sample(
                audio_file.replace(raw_data_path + "/", ""),
                text,
                get_length(audio_file),
            )

        samples = list(
            tqdm(
                process_with_threadpool(  # TODO(tilo): still not sure whether this is necessary
                    [{"audio_file": f, "text": t} for f, t in corpus.items()],
                    build_sample,
                    max_workers=10,
                )
            )
        )
        data_io.write_jsonl(file, (s._asdict() for s in samples))

    return samples


# fmt: off
LIBRI_VOCAB = [BLANK_SYMBOL, "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", SPACE]
# fmt: on


if __name__ == "__main__":
    datasets = [
        ("train", ["train-clean-100", "train-clean-360", "train-other-500"]),
        ("eval", ["dev-clean", "dev-other"]),
        ("test", ["test-clean", "test-other"]),
    ]
    start = time()
    for name, folders in datasets:
        samples = build_librispeech_corpus(
            HOME + "/data/asr_data/ENGLISH/LibriSpeech", name, folders, reprocess=True
        )
        print("%s got %d samples" % (name, len(samples)))

    print("took: %0.2f seconds" % (time() - start))

""":return
in %s/train-clean-100 found 28539 audio-files
in .../train-clean-360 found 104014 audio-files
in .../train-other-500 found 148688 audio-files
281241it [01:47, 2614.41it/s]
train got 281241 samples
in .../dev-clean found 2703 audio-files
in .../dev-other found 2864 audio-files
5567it [00:02, 2689.19it/s]
eval got 5567 samples
in .../test-clean found 2620 audio-files
in .../test-other found 2939 audio-files
5559it [00:02, 2710.73it/s]
test got 5559 samples
took: 127.06 seconds
"""
