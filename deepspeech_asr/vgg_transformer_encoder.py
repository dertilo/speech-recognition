import argparse
import math
from typing import Iterable, NamedTuple, List

import torch
from torch import nn as nn

from fairseq.modules import VGGBlock, TransformerEncoderLayer

"""
# based on fairseqs speech-recognition example
"""


def lengths_to_encoder_padding_mask(lengths, batch_first=False):
    """
    convert lengths (a 1-D Long/Int tensor) to 2-D binary tensor

    Args:
        lengths: a (B, )-shaped tensor

    Return:
        max_length: maximum length of B sequences
        encoder_padding_mask: a (max_length, B) binary mask, where
        [t, b] = 0 for t < lengths[b] and 1 otherwise

    TODO:
        kernelize this function if benchmarking shows this function is slow
        TODO(tilo): and it is super ugly!
    """
    max_lengths = torch.max(lengths).item()
    bsz = lengths.size(0)
    encoder_padding_mask = torch.arange(
        max_lengths
    ).to(  # a (T, ) tensor with [0, ..., T-1]
        lengths.device
    ).view(  # move to the right device
        1, max_lengths
    ).expand(  # reshape to (1, T)-shaped tensor
        bsz, -1
    ) >= lengths.view(  # expand to (B, T)-shaped tensor
        bsz, 1
    ).expand(
        -1, max_lengths
    )
    if not batch_first:
        return encoder_padding_mask.t(), max_lengths
    else:
        return encoder_padding_mask, max_lengths


class TransformerLayerConfig(NamedTuple):
    input_dim: int
    num_heads: int
    ffn_dim: int
    normalize_before: bool
    dropout: float
    attention_dropout: float
    relu_dropout: float


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


def validate_transformer_config(transformer_config:List[TransformerLayerConfig]):
    for config in transformer_config:
        input_dim, num_heads = config[:2]
        if input_dim % num_heads != 0:
            msg = (
                "ERROR in transformer config {}:".format(config)
                + "input dimension {} ".format(input_dim)
                + "not dividable by number of heads".format(num_heads)
            )
            raise ValueError(msg)


def Linear(in_features, out_features, bias=True):
    return nn.Linear(in_features, out_features, bias=bias)


def build_transformer_encoder(
    encoder_output_dim, tfcs: List[TransformerLayerConfig], transformer_input_dim
):
    validate_transformer_config(tfcs)
    layers = nn.ModuleList()
    if transformer_input_dim != tfcs[0].input_dim:
        layers.append(Linear(transformer_input_dim, tfcs[0].input_dim))
    layers.append(TransformerEncoderLayer(prepare_transformer_encoder_params(*tfcs[0])))
    for i in range(1, len(tfcs)):
        if tfcs[i - 1].input_dim != tfcs[i].input_dim:
            layers.append(Linear(tfcs[i - 1].input_dim, tfcs[i].input_dim))
        layers.append(
            TransformerEncoderLayer(prepare_transformer_encoder_params(*tfcs[i]))
        )
    layers.extend(
        [Linear(tfcs[-1].input_dim, encoder_output_dim), LayerNorm(encoder_output_dim),]
    )
    return layers


class VGGTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        input_feat_per_channel,
        vggblock_config,
        transformer_config,
        encoder_output_dim=512,
        in_channels=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel
        self.conv_layers, transformer_input_dim = build_vggblock(
            in_channels, input_feat_per_channel, vggblock_config
        )
        self.num_vggblocks = len(vggblock_config)

        self.encoder_output_dim = encoder_output_dim
        self.transformer_layers = build_transformer_encoder(
            encoder_output_dim,
            [TransformerLayerConfig(*p) for p in transformer_config],
            transformer_input_dim,
        )
        self.fc_out = nn.Linear(
            self.encoder_output_dim, vocab_size
        )  # bias=True TODO(tilo) why should this have a bias??

    def forward(self, src_tokens, src_lengths):
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

        for layer_idx in range(len(self.transformer_layers)):

            if isinstance(self.transformer_layers[layer_idx], TransformerEncoderLayer):
                x = self.transformer_layers[layer_idx](x, encoder_padding_mask)

            else:
                x = self.transformer_layers[layer_idx](x)

        # encoder_padding_maks is a (T x B) tensor, its [t, b] elements indicate
        # whether encoder_output[t, b] is valid or not (valid=0, invalid=1)

        probas = self.fc_out(x).transpose(1,0)
        return probas, input_lengths

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out
