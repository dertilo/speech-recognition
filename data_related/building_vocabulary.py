import os
from collections import Counter
from pprint import pprint

from tqdm import tqdm
from util import data_io

from utils import BLANK_SYMBOL


def build_vocabulary(
    corpus_file="spanish_train.jsonl",
    vocab_file="data/labels/vocabulary.json",
    min_freq=1000,
):
    text_g = (t for _, t in data_io.read_jsonl(corpus_file))
    counter = Counter((c.lower() for t in tqdm(text_g) for c in t))
    vocab = counter.most_common(200)
    data_io.write_json(
        vocab_file.replace(".json", "_freqs.json"),
        [(c, f) for c, f in vocab if f > min_freq],
    )
    data_io.write_json(vocab_file, [BLANK_SYMBOL] + [c for c, f in vocab if f > min_freq])


if __name__ == "__main__":

    jsonl = os.environ["HOME"] + "/data/asr_data/SPANISH/spanish_train.jsonl"
    build_vocabulary(jsonl, "spanish_vocab.json")
