import Levenshtein
from typing import List


def calc_num_word_errors(s1: str, s2: str) -> int:
    def tokenize(s):
        return s.split()

    b = set(tokenize(s1) + tokenize(s2))
    token2idx = {t: k for k, t in enumerate(b)}

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(token2idx[w]) for w in s1.split()]
    w2 = [chr(token2idx[w]) for w in s2.split()]
    return Levenshtein.distance("".join(w1), "".join(w2))


def calc_num_char_erros(s1: str, s2: str):
    s1, s2, = (
        s1.replace(" ", ""),
        s2.replace(" ", ""),
    )  # TODO(tilo): why removing spaces?
    return Levenshtein.distance(s1, s2)


def calc_wer(hypos: List[str], targets: List[str]):
    errors_lens = [
        (calc_num_word_errors(hyp, ref), len(ref.split(" ")))
        for hyp, ref in zip(hypos, targets)
    ]
    num_tokens = sum([l for _, l in errors_lens])
    errors = sum([s for s, _ in errors_lens])
    wer = float(errors)/float(num_tokens)
    return wer

def calc_cer(hypos: List[str], targets: List[str]):
    errors_lens = [
        (calc_num_char_erros(hyp, ref), len(ref.replace(' ','')))
        for hyp, ref in zip(hypos, targets)
    ]
    num_tokens = sum([l for _, l in errors_lens])
    errors = sum([s for s, _ in errors_lens])
    cer = float(errors)/float(num_tokens)
    return cer
