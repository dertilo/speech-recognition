import Levenshtein


def calc_num_word_errors(s1:str, s2:str)->int:
    def tokenize(s):
        return s.split()

    b = set(tokenize(s1) + tokenize(s2))
    token2idx = {t: k for k, t in enumerate(b)}

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(token2idx[w]) for w in s1.split()]
    w2 = [chr(token2idx[w]) for w in s2.split()]
    return Levenshtein.distance("".join(w1), "".join(w2))

def calc_num_char_erros(s1:str, s2:str):
    s1, s2, = s1.replace(" ", ""), s2.replace(" ", "") # TODO(tilo): why removing spaces?
    return Levenshtein.distance(s1, s2)
