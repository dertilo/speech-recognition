import argparse
import os
from abc import abstractmethod
from typing import NamedTuple, Dict, Union, List

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from test_tube import HyperOptArgumentParser
from collections import OrderedDict
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader

from decoder import Decoder, convert_to_strings
from lightning.litutil import add_generic_args, build_args
from metrics_calculation import calc_num_word_errors, calc_num_char_erros
from transcribing.transcribe_util import build_decoder
from utils import BLANK_SYMBOL


def collate(batch):
    batch = sorted(
        batch, key=lambda sample: sample[0].size(1), reverse=True
    )  # why? cause "nn.utils.rnn.pack_padded_sequence" want it like this!
    inputs, targets = [list(x) for x in zip(*batch)]
    target_sizes = torch.LongTensor([len(t) for t in targets])
    targets = [torch.IntTensor(target) for target in targets]
    padded_target = pad_sequence(targets, batch_first=True)
    input_sizes = torch.LongTensor([x.size(1) for x in inputs])
    padded_inputs = pad_sequence([i.transpose(1, 0) for i in inputs], batch_first=True)
    return padded_inputs, padded_target, input_sizes, target_sizes


class LitSTTModel(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams.lr
        self.model = self._build_model(hparams)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.decoder = build_decoder(self.char2idx, use_beam_decoder=False)
        self.BLANK_INDEX = self.char2idx[BLANK_SYMBOL]

    @property
    @abstractmethod
    def char2idx(self):
        raise NotImplementedError

    @abstractmethod
    def _build_model(self, hparams):
        raise NotImplementedError

    def forward(self, inputs, input_sizes):
        return self.model(inputs, input_sizes)

    # TODO(tilo): this must be unused! cause its not working!
    # def decode(self, feature, feature_length, decode_type="greedy"):
    #     assert decode_type in ["greedy", "beam"]
    #     output = self.transformer.inference(
    #         feature, feature_length, decode_type=decode_type
    #     )
    #     return output

    def training_step(self, batch, batch_nb):

        inputs, targets, input_sizes, target_sizes = batch

        out, output_sizes = self(inputs, input_sizes)

        loss = self.calc_loss(out, output_sizes, targets, target_sizes)
        tqdm_dict = {"train-loss": loss.item()}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict,}
        )
        return output

    @abstractmethod
    def _supply_trainset(self):
        raise NotImplementedError

    @abstractmethod
    def _supply_evalset(self):
        raise NotImplementedError

    def calc_loss(
        self, out, output_sizes, targets, target_sizes,
    ):
        prob = F.log_softmax(out, -1)
        ctc_loss = F.ctc_loss(
            prob.transpose(0, 1),
            targets,
            output_sizes,
            target_sizes,
            blank=self.BLANK_INDEX,
            reduction="sum",
            zero_infinity=True,
        )

        batch_size = out.size(0)
        loss = ctc_loss / batch_size  # average the loss by minibatch
        return loss

    def _calc_error(self, targets, decoded_output):
        target_strings = convert_to_strings(
            self.idx2char, self.char2idx[BLANK_SYMBOL], targets
        )
        total_wer, total_cer, num_chars, num_tokens = 0, 0, 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            total_wer += calc_num_word_errors(transcript, reference)
            total_cer += calc_num_char_erros(transcript, reference)
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(" ", ""))
        return total_wer, total_cer, num_tokens, num_chars

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> Dict:
        inputs, targets, input_sizes, target_sizes = batch

        decoded_output, out, output_sizes = transcribe_batch(
            self.decoder, input_sizes, inputs, self.model
        )
        loss_value = self.calc_loss(out, output_sizes, targets, target_sizes).item()
        total_wer, total_cer, num_tokens, num_chars = self._calc_error(
            targets, decoded_output
        )

        tqdm_dict = {
            "val-loss": loss_value,
        }
        output = OrderedDict(
            {"loss": loss_value, "progress_bar": tqdm_dict, "log": tqdm_dict,}
        )
        output.update(
            {
                "wer": total_wer,
                "cer": total_cer,
                "num_tokens": num_tokens,
                "num_chars": num_chars,
            }
        )
        return output

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict[str, Dict[str, Tensor]]:

        sums = outputs[0]
        for d in outputs[1:]:
            for k, v in d.items():
                if k not in ["log", "progress_bar"]:
                    sums[k] += v

        avg_wer = sums["wer"] / sums["num_tokens"]
        avg_cer = sums["cer"] / sums["num_chars"]
        val_loss_mean = sums["loss"] / len(outputs)
        tqdm_dict = {
            "val_loss": val_loss_mean,
            "wer": avg_wer,
            "cer": avg_cer,
        }
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            "wer": avg_wer,
            "cer": avg_cer,
        }
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        return optimizer

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser])
        parser.add_argument("--lr", default=3e-4, type=float)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--vocab_size", type=int)
        parser.add_argument("--audio_feature_dim", type=int)
        return parser


def transcribe_batch(decoder: Decoder, input_sizes, inputs, model):
    out, output_sizes = model(inputs, input_sizes)
    probs = F.softmax(out, dim=-1)
    decoded_output, _ = decoder.decode(probs, output_sizes)
    return decoded_output, out, output_sizes
