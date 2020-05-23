import os
from typing import NamedTuple, Dict, Union, List

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser
from collections import OrderedDict
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader

from data_related.librispeech import LIBRI_VOCAB, build_dataset
from decoder import Decoder, convert_to_strings
from metrics_calculation import calc_num_word_errors, calc_num_char_erros
from model import DeepSpeech
from transcribing.transcribe_util import build_decoder
from utils import BLANK_SYMBOL


class Params(NamedTuple):
    hidden_size: int
    hidden_layers: int
    audio_feature_dim: int
    vocab_size: int
    bidirectional: bool = True
    num_workers: int = 0
    batch_size: int = 4
    lr: float = 0.0003


class LitSTTModel(pl.LightningModule):
    def __init__(self, hparams: Params):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams.lr
        self.model = DeepSpeech(
            hidden_size=hparams.hidden_size,
            nb_layers=hparams.hidden_layers,
            vocab_size=hparams.vocab_size,
            input_feature_dim=hparams.audio_feature_dim,
            bidirectional=hparams.bidirectional,
        )
        self.char2idx = dict([(l, i) for i, l in enumerate(LIBRI_VOCAB)])
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.decoder = build_decoder(self.char2idx, use_beam_decoder=False)
        self.BLANK_INDEX = self.char2idx[BLANK_SYMBOL]

    def forward(self, inputs, input_sizes):
        return self.model(inputs, input_sizes)

    def decode(self, feature, feature_length, decode_type="greedy"):
        assert decode_type in ["greedy", "beam"]
        output = self.transformer.inference(
            feature, feature_length, decode_type=decode_type
        )
        return output

    def training_step(self, batch, batch_nb):

        inputs, targets, input_sizes, target_sizes = batch

        out, output_sizes = self(inputs, input_sizes)

        loss = self.calc_loss(out, output_sizes, targets, target_sizes)
        tqdm_dict = {"train-loss": loss.item()}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict,}
        )
        return output

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataset = build_dataset(
            "train-100",
            ["train-clean-100"]#, "train-clean-360", "train-other-500"]
            # "debug",
            # ["dev-clean"],
        )
        dataloader = DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=_collate_fn,
        )
        return dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataset = build_dataset("eval", ["dev-clean", "dev-other"])
        dataloader = DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=_collate_fn,
        )
        return dataloader

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser])
        parser.add_argument("--num_encoder_layer", default=6, type=int)
        parser.add_argument("--lr", default=3e-4, type=float)
        parser.add_argument("--train_batch_size", default=20, type=int)
        parser.add_argument("--val_batch_size", default=20, type=int)
        return parser


def transcribe_batch(decoder: Decoder, input_sizes, inputs, model):
    out, output_sizes = model(inputs, input_sizes)
    probs = F.softmax(out, dim=-1)
    decoded_output, _ = decoder.decode(probs, output_sizes)
    return decoded_output, out, output_sizes


def _collate_fn(batch):
    batch = sorted(
        batch, key=lambda sample: sample[0].size(1), reverse=True
    )  # why? cause "nn.utils.rnn.pack_padded_sequence" want it like this!
    inputs, targets = [list(x) for x in zip(*batch)]
    target_sizes = torch.LongTensor([len(t) for t in targets])
    targets = [torch.IntTensor(target) for target in targets]
    padded_target = pad_sequence(targets, batch_first=True)
    input_sizes = torch.LongTensor([x.size(1) for x in inputs])
    padded_inputs = pad_sequence([i.transpose(1, 0) for i in inputs], batch_first=True)
    padded_inputs = padded_inputs.unsqueeze(1).transpose(
        3, 2
    )  # DeepSpeech wants it like this
    return padded_inputs, padded_target, input_sizes, target_sizes


if __name__ == "__main__":

    data_path = os.environ["HOME"] + "/data/asr_data/"

    train_dataset = build_dataset()
    vocab_size = len(train_dataset.char2idx)
    BLANK_INDEX = train_dataset.char2idx[BLANK_SYMBOL]
    audio_feature_dim = train_dataset.audio_fe.feature_dim

    model = LitSTTModel(
        Params(
            hidden_size=64,
            hidden_layers=2,
            audio_feature_dim=audio_feature_dim,
            vocab_size=vocab_size,
            batch_size=4,
            num_workers=0,
        )
    )
    trainer = Trainer(logger=False, checkpoint_callback=False, max_epochs=3)
    trainer.fit(model)
