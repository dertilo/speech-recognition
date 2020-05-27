import os
from warnings import filterwarnings

import torch
from torch.nn.utils.rnn import pad_sequence

from data_related.librispeech import build_dataset, LIBRI_VOCAB
from lightning.lightning_model import LitSTTModel
from lightning.litutil import generic_train, build_args
from vgg_transformer_encoder import VGGTransformerEncoder

filterwarnings("ignore")


class LitVGGTransformerEncoder(LitSTTModel):
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
        return VGGTransformerEncoder(
            vocab_size=args.vocab_size,
            input_feat_per_channel=args.input_feat_per_channel,
            vggblock_config=eval(args.vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.enc_output_dim,
            in_channels=args.in_channels,
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--input_feat_per_channel", default=40, type=int)
        parser.add_argument("--enc_output_dim", default=512, type=int)
        parser.add_argument(
            "--vggblock_enc_config", default="[(32, 3, 2, 2, True)] * 2", type=str
        )
        parser.add_argument(
            "--transformer_enc_config",
            default="((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2",
            type=str,
        )
        parser.add_argument("--in_channels", default=1, type=int)
        return parser

    @staticmethod
    def _collate_fn(batch):
        batch = sorted(
            batch, key=lambda sample: sample[0].size(1), reverse=True
        )  # why? cause "nn.utils.rnn.pack_padded_sequence" want it like this!
        inputs, targets = [list(x) for x in zip(*batch)]
        target_sizes = torch.LongTensor([len(t) for t in targets])
        targets = [torch.IntTensor(target) for target in targets]
        padded_target = pad_sequence(targets, batch_first=True)
        input_sizes = torch.LongTensor([x.size(1) for x in inputs])
        padded_inputs = pad_sequence(
            [i.transpose(1, 0) for i in inputs], batch_first=True
        )
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
        "enc_output_dim": 64,
        "transformer_enc_config": "((32, 4, 128, True, 0.2, 0.2, 0.2),) * 2",
        "num_workers": 0,
        "max_epochs": 1,
    }
    args = build_args(LitVGGTransformerEncoder, p)

    train_dataset = build_dataset()
    args.vocab_size = len(train_dataset.char2idx)
    # BLANK_INDEX = train_dataset.char2idx[BLANK_SYMBOL]
    args.input_feat_per_channel = train_dataset.audio_fe.feature_dim

    model = LitVGGTransformerEncoder(args)

    generic_train(model, args)
