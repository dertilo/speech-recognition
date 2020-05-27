import os
from warnings import filterwarnings

import torch
from torch.nn.utils.rnn import pad_sequence

from data_related.librispeech import build_dataset, LIBRI_VOCAB
from lightning.lightning_model import LitSTTModel, collate
from lightning.litutil import generic_train, build_args
from model import DeepSpeech

filterwarnings("ignore")


class LitDeepSpeech(LitSTTModel):
    def _supply_trainset(self):  # TODO(tilo) should this be an argument??
        dataset = build_dataset(
            "train-100",
            ["train-clean-100"]  # , "train-clean-360", "train-other-500"]
            # "debug",
            # ["dev-clean"],
        )
        dataset.samples = dataset.samples[:100]
        return dataset

    @property
    def char2idx(self):
        return dict([(l, i) for i, l in enumerate(LIBRI_VOCAB)])

    def _supply_evalset(self):
        dataset = build_dataset("eval", ["dev-clean", "dev-other"])
        dataset.samples = dataset.samples[:100]
        return dataset

    def _build_model(self, hparams):
        return DeepSpeech(
            hidden_size=hparams.hidden_size,
            nb_layers=hparams.hidden_layers,
            vocab_size=hparams.vocab_size,
            input_feature_dim=hparams.audio_feature_dim,
            bidirectional=hparams.bidirectional,
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--bidirectional", default=True, type=bool)
        parser.add_argument("--hidden_layers", default=5, type=int)
        parser.add_argument("--hidden_size", default=1024, type=int)
        return parser

    @classmethod
    def _collate_fn(cls, batch):
        padded_inputs, padded_target, input_sizes, target_sizes = super()._collate_fn(
            batch
        )
        padded_inputs = padded_inputs.unsqueeze(1).transpose(
            3, 2
        )  # DeepSpeech wants it like this
        return padded_inputs, padded_target, input_sizes, target_sizes


if __name__ == "__main__":
    data_path = os.environ["HOME"] + "/data/asr_data/"
    p = {
        # "exp_name": "deepspeech-train-100",
        "exp_name": "debug",
        "run_name": "some run",
        "save_path": data_path + "/mlruns",
        "batch_size": 4,
        # "fp16": "bla",
        "n_gpu": 0,
        "hidden_layers": 2,
        "hidden_size": 64,
        "num_workers": 0,
        "max_epochs": 1,
    }
    args = build_args(LitDeepSpeech, p)

    train_dataset = build_dataset()
    args.vocab_size = len(train_dataset.char2idx)
    # BLANK_INDEX = train_dataset.char2idx[BLANK_SYMBOL]
    args.audio_feature_dim = train_dataset.audio_fe.feature_dim

    model = LitDeepSpeech(args)

    generic_train(model, args)
