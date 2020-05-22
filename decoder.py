#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch
from six.moves import xrange
from typing import Dict, NamedTuple

from utils import BLANK_SYMBOL, SPACE


class Decoder(object):
    def __init__(self, char2idx: Dict[str, int]):
        self.char2idx = char2idx
        self.idx2char = {v: k for k, v in char2idx.items()}
        self.blank_index = char2idx[BLANK_SYMBOL]
        self.space_index = char2idx[SPACE]

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class DecoderConfig(NamedTuple):
    lm_path: str = None
    alpha: float = 0
    beta: float = 0
    cutoff_top_n: int = 40
    cutoff_prob: float = 1.0
    beam_width: int = 10
    num_processes: int = 4
    blank_index: int = 0


class BeamCTCDecoder(Decoder):
    def __init__(
        self,
        char2idx,
        lm_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_index=0,
    ):
        super(BeamCTCDecoder, self).__init__(char2idx)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        self._decoder = CTCBeamDecoder(
            char2idx,
            lm_path,
            alpha,
            beta,
            cutoff_top_n,
            cutoff_prob,
            beam_width,
            num_processes,
            blank_index,
        )

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = "".join(
                        map(lambda x: self.idx2char[x.item()], utt[0:size])
                    )
                else:
                    transcript = ""
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


def process_string(idx2char, blank_index,sequence,size, remove_repetitions=False):
    string = ""
    offsets = []
    for i in range(size):
        char = idx2char[sequence[i].item()]
        if char != idx2char[blank_index]:
            # if this char is a repetition and remove_repetitions=true, then skip
            if (
                remove_repetitions
                and i != 0
                and char == idx2char[sequence[i - 1].item()]
            ):
                pass
            elif char == SPACE:
                string += " "
                offsets.append(i)
            else:
                string = string + char
                offsets.append(i)
    return string, torch.tensor(offsets, dtype=torch.int)


def convert_to_strings(
    idx2char, blank_index,sequences, remove_repetitions=False, return_offsets=False, sizes=None
):
    """Given a list of numeric sequences, returns the corresponding strings"""
    strings = []
    offsets = [] if return_offsets else None
    for x in xrange(len(sequences)):
        seq_len = sizes[x] if sizes is not None else len(sequences[x])
        string, string_offsets = process_string(
            idx2char, blank_index, sequences[x], seq_len, remove_repetitions
        )
        strings.append([string])  # We only return one path
        if return_offsets:
            offsets.append([string_offsets])
    if return_offsets:
        return strings, offsets
    else:
        return strings


class GreedyDecoder(Decoder):
    def __init__(self, char2idx):
        super(GreedyDecoder, self).__init__(char2idx)

    def convert_to_strings(
        self, sequences, sizes=None, remove_repetitions=False, return_offsets=False
    ):
        return convert_to_strings(
            self.idx2char,
            self.blank_index,
            sequences,
            remove_repetitions,
            return_offsets,
            sizes,
        )

    def process_string(self, sequence, size, remove_repetitions=False):
        return process_string(
            self.idx2char, self.blank_index, sequence, size,remove_repetitions
        )

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(
            max_probs.view(max_probs.size(0), max_probs.size(1)),
            sizes,
            remove_repetitions=True,
            return_offsets=True,
        )
        return strings, offsets
