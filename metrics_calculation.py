import Levenshtein


def calc_wer(s1:str, s2:str):
    def tokenize(s):
        return s.split()

    b = set(tokenize(s1) + tokenize(s2))
    token2idx = {k: s for k, s in enumerate(b)}

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(token2idx[w]) for w in s1.split()]
    w2 = [chr(token2idx[w]) for w in s2.split()]
    return Levenshtein.distance("".join(w1), "".join(w2))

def calc_cer(s1:str, s2:str):
    s1, s2, = s1.replace(" ", ""), s2.replace(" ", "") # TODO(tilo): why removing spaces?
    return Levenshtein.distance(s1, s2)
