import argparse
import math
from typing import Iterable

import torch
from torch import nn as nn

from fairseq.models import FairseqEncoder
from fairseq.modules import VGGBlock, TransformerEncoderLayer
from speech_recognition.data.data_utils import lengths_to_encoder_padding_mask
from speech_recognition.models.asr_models_common import Linear

""":arg
# based on fairseqs speech-recognition example
"""


def prepare_transformer_encoder_params(
    input_dim,
    num_heads,
    ffn_dim,
    normalize_before,
    dropout,
    attention_dropout,
    relu_dropout,
):
    args = argparse.Namespace()
    args.encoder_embed_dim = input_dim
    args.encoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.encoder_normalize_before = normalize_before
    args.encoder_ffn_embed_dim = ffn_dim
    return args


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def add_encoder_args(parser):
    parser.add_argument(
        "--input-feat-per-channel",
        type=int,
        metavar="N",
        help="encoder input dimension per input channel",
    )
    parser.add_argument(
        "--vggblock-enc-config",
        type=str,
        metavar="EXPR",
        help="""
        an array of tuples each containing the configuration of one vggblock:
        [(out_channels,
          conv_kernel_size,
          pooling_kernel_size,
          num_conv_layers,
          use_layer_norm), ...])
            """,
    )
    parser.add_argument(
        "--transformer-enc-config",
        type=str,
        metavar="EXPR",
        help=""""
        a tuple containing the configuration of the encoder transformer layers
        configurations:
        [(input_dim,
          num_heads,
          ffn_dim,
          normalize_before,
          dropout,
          attention_dropout,
          relu_dropout), ...]')
                """,
    )
    parser.add_argument(
        "--enc-output-dim",
        type=int,
        metavar="N",
        help="""
        encoder output dimension, can be None. If specified, projecting the
        transformer output to the specified dimension""",
    )
    parser.add_argument(
        "--in-channels", type=int, metavar="N", help="number of encoder input channels",
    )


DEFAULT_ENC_VGGBLOCK_CONFIG = ((32, 3, 2, 2, False),) * 2
DEFAULT_ENC_TRANSFORMER_CONFIG = ((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2
# 256: embedding dimension
# 4: number of heads
# 1024: FFN
# True: apply layerNorm before (dropout + resiaul) instead of after
# 0.2 (dropout): dropout after MultiheadAttention and second FC
# 0.2 (attention_dropout): dropout in MultiheadAttention
# 0.2 (relu_dropout): dropout after ReLu


def build_vggblock(in_channels, input_feat_per_channel, vggblock_config):
    input_dim = input_feat_per_channel
    inin_channels = in_channels
    conv_layers = nn.ModuleList()
    if vggblock_config is not None:
        for (
            out_channels,
            conv_kernel_size,
            pooling_kernel_size,
            num_conv_layers,
            layer_norm,
        ) in vggblock_config:
            conv_layers.append(
                VGGBlock(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    pooling_kernel_size,
                    num_conv_layers,
                    input_dim=input_feat_per_channel,
                    layer_norm=layer_norm,
                )
            )
            in_channels = out_channels
            input_feat_per_channel = conv_layers[-1].output_dim

    def infer_conv_output_dim(in_channels, input_dim):  # TODO(tilo): WTF!
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        for i, _ in enumerate(conv_layers):
            x = conv_layers[i](x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    transformer_input_dim = infer_conv_output_dim(inin_channels, input_dim)
    return conv_layers, transformer_input_dim

def validate_transformer_config(transformer_config):
    for config in transformer_config:
        input_dim, num_heads = config[:2]
        if input_dim % num_heads != 0:
            msg = (
                "ERROR in transformer config {}:".format(config)
                + "input dimension {} ".format(input_dim)
                + "not dividable by number of heads".format(num_heads)
            )
            raise ValueError(msg)

class VGGTransformerEncoder(FairseqEncoder):
    """VGG + Transformer encoder"""

    def __init__(
        self,
        input_feat_per_channel,
        vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
        transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
        encoder_output_dim=512,
        in_channels=1,
        transformer_context=None,
        transformer_sampling=None,
    ):
        """constructor for VGGTransformerEncoder

        Args:
            - input_feat_per_channel: feature dim (not including stacked,
              just base feature)
            - in_channel: # input channels (e.g., if stack 8 feature vector
                together, this is 8)
            - vggblock_config: configuration of vggblock, see comments on
                DEFAULT_ENC_VGGBLOCK_CONFIG
            - transformer_config: configuration of transformer layer, see comments
                on DEFAULT_ENC_TRANSFORMER_CONFIG
            - encoder_output_dim: final transformer output embedding dimension
            - transformer_context: (left, right) if set, self-attention will be focused
              on (t-left, t+right)
            - transformer_sampling: an iterable of int, must match with
              len(transformer_config), transformer_sampling[i] indicates sampling
              factor for i-th transformer layer, after multihead att and feedfoward
              part
        """
        assert transformer_context is None # TODO(tilo) what are they good for?
        assert transformer_sampling is None
        super().__init__(None)

        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel
        self.conv_layers, transformer_input_dim = build_vggblock(
            in_channels, input_feat_per_channel, vggblock_config
        )
        self.num_vggblocks = len(vggblock_config)

        # transformer_input_dim is the output dimension of VGG part

        validate_transformer_config(transformer_config)
        self.transformer_sampling = (1,) * len(transformer_config)

        self.transformer_layers = nn.ModuleList()

        if transformer_input_dim != transformer_config[0][0]:
            self.transformer_layers.append(
                Linear(transformer_input_dim, transformer_config[0][0])
            )
        self.transformer_layers.append(
            TransformerEncoderLayer(
                prepare_transformer_encoder_params(*transformer_config[0])
            )
        )

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.transformer_layers.append(
                    Linear(transformer_config[i - 1][0], transformer_config[i][0])
                )
            self.transformer_layers.append(
                TransformerEncoderLayer(
                    prepare_transformer_encoder_params(*transformer_config[i])
                )
            )

        self.encoder_output_dim = encoder_output_dim
        self.transformer_layers.extend(
            [
                Linear(transformer_config[-1][0], encoder_output_dim),
                LayerNorm(encoder_output_dim),
            ]
        )

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        x = x.transpose(1, 2).contiguous()
        # (B, C, T, feat)

        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)

        bsz, _, output_seq_len, _ = x.size()

        # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) -> (T, B, C * feat)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.contiguous().view(output_seq_len, bsz, -1)

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        # TODO: shouldn't subsampling_factor determined in advance ?
        input_lengths = (src_lengths.float() / subsampling_factor).ceil().long()

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        attn_mask = None

        transformer_layer_idx = 0

        for layer_idx in range(len(self.transformer_layers)):

            if isinstance(self.transformer_layers[layer_idx], TransformerEncoderLayer):
                x = self.transformer_layers[layer_idx](
                    x, encoder_padding_mask, attn_mask
                )

                transformer_layer_idx += 1

            else:
                x = self.transformer_layers[layer_idx](x)

        # encoder_padding_maks is a (T x B) tensor, its [t, b] elements indicate
        # whether encoder_output[t, b] is valid or not (valid=0, invalid=1)

        return {
            "encoder_out": x,  # (T, B, C)
            "encoder_padding_mask": encoder_padding_mask.t()
            if encoder_padding_mask is not None
            else None,
            # (B, T) --> (T, B)
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out
