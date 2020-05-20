from __future__ import absolute_import, division, print_function, unicode_literals

import os
from typing import List
from tqdm import tqdm
from util import data_io
from util.util_methods import process_with_threadpool
from corpora.librispeech import librispeech_corpus
from data_related.audio_feature_extraction import get_length
from data_related.audio_util import Sample
from utils import HOME, BLANK_SYMBOL, SPACE


def load_samples(file: str, base_path: str) -> List[Sample]:
    def process(d):
        split_str = "asr_data/"
        if split_str in d["audio_file"]:  # TODO(tilo):only for backward compatibility
            _, s = d["audio_file"].split(split_str)
            p, _ = base_path.split(split_str)
            file = os.path.join(p, split_str, s)
        else:
            file = os.path.join(base_path, d["audio_file"])
        d["audio_file"] = file
        return Sample(**d)

    return [process(d) for d in data_io.read_jsonl(file)]


def build_librispeech_corpus(
    raw_data_path, name: str, folders: List[str]
) -> List[Sample]:
    file = raw_data_path + "/%s_samples.jsonl.gz" % name

    if os.path.isfile(file):
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
            return Sample(audio_file, text, get_length(audio_file))

        samples = list(
            tqdm(
                process_with_threadpool(
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
    for name, folders in datasets:
        samples = build_librispeech_corpus(
            HOME + "/data/asr_data/ENGLISH/LibriSpeech", name, folders
        )
        print("%s got %d samples" % (name, len(samples)))