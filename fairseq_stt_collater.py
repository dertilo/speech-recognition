"""
    based on:  https://github.com/pytorch/fairseq/examples/speech_recognition/data/collaters.py
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import torch
from typing import Dict, List, NamedTuple

from fairseq.data import data_utils as fairseq_data_utils


class Sample(NamedTuple):
    id: int
    source: torch.Tensor
    target: torch.Tensor = None


def _sort_frames_by_lengths(frames, id, samples):
    frames_lengths = torch.LongTensor([s.source.size(0) for s in samples])
    frames_lengths, sort_order = frames_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    frames = frames.index_select(0, sort_order)
    return frames, frames_lengths, id, sort_order


def pad_and_concat_frames(frames):
    """Convert a list of 2d frames into a padded 3d tensor
    Args:
        frames (list): list of 2d frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3d tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    len_max = max(frame.size(0) for frame in frames)
    f_dim = frames[0].size(1)
    res = frames[0].new(len(frames), len_max, f_dim).fill_(0.0)

    for i, v in enumerate(frames):
        res[i, : v.size(0)] = v

    return res


def collate_target_tokens(samples, sort_order, do_rightshift, eos_index, pad_index):
    target = fairseq_data_utils.collate_tokens(
        [s.target for s in samples],
        pad_index,
        eos_index,
        left_pad=False,
        move_eos_to_beginning=do_rightshift,
    )
    target = target.index_select(0, sort_order)
    return target


def prepare_targets(samples: List[Sample], sort_order, pad_index, eos_index):
    ntokens = sum(len(s.target) for s in samples)
    target = collate_target_tokens(samples, sort_order, False, eos_index, pad_index)
    prev_output_tokens = collate_target_tokens(
        samples, sort_order, True, eos_index, pad_index
    )

    target_lengths = torch.LongTensor([s.target.size(0) for s in samples]).index_select(
        0, sort_order
    )

    return ntokens, prev_output_tokens, target, target_lengths


def collate(samples: List[Sample], pad_index, eos_index):
    id = torch.LongTensor([s.id for s in samples])
    frames = pad_and_concat_frames([s.source for s in samples])
    frames, frames_lengths, id, sort_order = _sort_frames_by_lengths(
        frames, id, samples
    )
    if samples[0].target is not None:
        ntokens, prev_output_tokens, target, target_lengths = prepare_targets(
            samples, sort_order, pad_index, eos_index
        )
    else:
        target = None
        target_lengths = None
        prev_output_tokens = None
        ntokens = sum(len(s.source) for s in samples)
    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {"src_tokens": frames, "src_lengths": frames_lengths},
        "target": target,
        "target_lengths": target_lengths,
        "nsentences": len(samples),
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens
    return batch


class Seq2SeqCollater(object):
    """
        Implements collate function mainly for seq2seq tasks
        This expects each sample to contain feature (src_tokens) and
        targets.
        This collator is also used for aligned training task.
    """

    def __init__(
        self,
        feature_index=0,
        label_index=1,
        pad_index=1,
        eos_index=2,
        move_eos_to_beginning=None,
    ):
        self.pad_index = pad_index
        self.eos_index = eos_index

    def collate(self, samples: List[Dict]):
        assert len(samples) > 0
        # if len(samples) == 0:
        #     return {}
        samples = [Sample(**s) for s in samples]
        return collate(samples, self.pad_index, self.eos_index)
