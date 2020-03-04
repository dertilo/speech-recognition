import os
from collections import Counter
from pprint import pprint

from tqdm import tqdm

from data_related.data_utils import read_jsonl, write_json

BLANK_CHAR = '_'

def build_vocabulary(
    corpus_file="spanish_train.jsonl",
    vocab_file="data/labels/vocabulary.json",
    min_freq=1000
):
    text_g = (t for _, t in read_jsonl(corpus_file))
    counter = Counter((c.lower() for t in tqdm(text_g) for c in t))
    vocab = counter.most_common(200)
    write_json(vocab_file.replace('.json','_freqs.json'),[(c,f) for c, f in vocab if f > min_freq])
    write_json(vocab_file, [BLANK_CHAR]+[c for c, f in vocab if f > min_freq])


if __name__ == "__main__":

    jsonl = os.environ['HOME']+'/data/asr_data/SPANISH/spanish_train.jsonl'
    build_vocabulary(jsonl,'spanish_vocab.json')
