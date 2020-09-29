import shutil
from dataclasses import dataclass, field

import yaml
from typing import Optional, Union, List

import os

import argparse

from espnet2.bin.tokenize_text import tokenize, get_parser
from espnet2.tasks.asr import ASRTask
from util import data_io, util_methods
import sentencepiece as spm
import shlex

TRAIN = "train"
VALID = "valid"
MANIFESTS = "manifests"
STATS = "stats"
TRAINLOGS = "train_logs"
TOKENIZER = "tokenizer"


def build_config(args):
    return f"""
    batch_type: numel
    batch_bins: 16_00_000
    accum_grad: 1
    max_epoch: 1
    patience: none
    # The initialization method for model parameters
    init: xavier_uniform
    best_model_criterion:
    -   - valid
        - acc
        - max
    keep_nbest_models: 10
    
    encoder: transformer
    encoder_conf:
        output_size: 32
        attention_heads: 2
        linear_units: 128
        num_blocks: {args.num_encoder_blocks}
        dropout_rate: 0.1
        positional_dropout_rate: 0.1
        attention_dropout_rate: 0.1
        input_layer: conv2d6
        normalize_before: true
    
    decoder: transformer
    decoder_conf:
        attention_heads: 2
        linear_units: 128
        num_blocks: 1
        dropout_rate: 0.1
        positional_dropout_rate: 0.1
        self_attention_dropout_rate: 0.1
        src_attention_dropout_rate: 0.1
    
    model_conf:
        ctc_weight: 0.3
        lsm_weight: 0.1
        length_normalized_loss: false
    
    optim: adam
    optim_conf:
        lr: 0.0
    scheduler: warmuplr
    scheduler_conf:
        warmup_steps: 25000
    
    specaug: specaug
    specaug_conf:
        apply_time_warp: true
        time_warp_window: 5
        time_warp_mode: bicubic
        apply_freq_mask: true
        freq_mask_width_range:
        - 0
        - 30
        num_freq_mask: 2
        apply_time_mask: true
        time_mask_width_range:
        - 0
        - 40
        num_time_mask: 2
    """


def build_manifest_files(
    manifest_path="/tmp",
    dataset_path="some-wehre/dev-clean_preprocessed",
    limit=None,  # just for debug
):
    os.makedirs(manifest_path, exist_ok=True)
    manifest_file = f"{dataset_path}/manifest.jsonl.gz"
    g = data_io.read_jsonl(manifest_file, limit=limit)
    data_io.write_lines(
        f"{manifest_path}/wav.scp",
        (
            f"{d['audio_file'].replace('.mp3', '')}\t{dataset_path}/{d['audio_file']}"
            for d in g
        ),
    )
    g = data_io.read_jsonl(manifest_file, limit=limit)
    data_io.write_lines(
        f"{manifest_path}/text",
        (f"{d['audio_file'].replace('.mp3', '')}\t{d['text']}" for d in g),
    )
    data_io.write_file(f"{manifest_path}/feats_type", "raw")


def write_vocabulary(tokenizer_dir="data/token_list/bpe_unigram5000"):
    cmd = (
        f"--token_type bpe --input {tokenizer_dir}/train.txt --output {tokenizer_dir}/tokens.txt "
        f"--bpemodel {tokenizer_dir}/bpe.model --field 2- --cleaner none --g2p none "
        f"--write_vocabulary true "
        f"--add_symbol '<blank>:0' --add_symbol '<unk>:1' --add_symbol '<sos/eos>:-1'"
    )
    parser = get_parser()
    args = parser.parse_args(shlex.split(cmd))
    kwargs = vars(args)
    tokenize(**kwargs)


def train_tokenizer(out_path, vocab_size=5000):
    td = f"{out_path}/{TOKENIZER}"

    os.makedirs(td, exist_ok=True)
    spm_args = dict(
        input=(f"{td}/train.txt"),
        vocab_size=vocab_size,
        model_type="unigram",
        model_prefix=(f"{td}/bpe"),
        character_coverage=1.0,
        input_sentence_size=100000000,
    )
    os.system(
        f'<"{out_path}/{MANIFESTS}/{TRAIN}/text" cut -f 2- -d" "  > {td}/train.txt'
    )

    spm.SentencePieceTrainer.Train(
        " ".join([f"--{k}={v}" for k, v in spm_args.items()])
    )

    write_vocabulary(td)


def run_asr_task(
    output_path,
    config,
    bpe_model: str,
    token_list: Union[str, List[str]],
    num_gpus=0,
    is_distributed=False,
    num_workers=0,
    pretrain_config=None,
    collect_stats=False,
):
    sp = f"{output_path}/{STATS}"

    output_dir = sp if collect_stats else f"{output_path}/{TRAINLOGS}"
    mp = f"{output_path}/{MANIFESTS}"
    argString = (
        f"--collect_stats {collect_stats} "
        f"--use_preprocessor true "
        f"--bpemodel {bpe_model} "
        f"--seed 42 "
        f"--num_workers {num_workers} "
        f"--token_type bpe "
        f"--token_list {token_list} "
        f"--g2p none "
        f"--non_linguistic_symbols none "
        f"--cleaner none "
        f"--resume true "
        f"--fold_length 5000 "
        f"--fold_length 150 "
        f"--config {config} "
        f"--frontend_conf fs=16k "
        f"--output_dir {output_dir} "
        f"--train_data_path_and_name_and_type {mp}/{TRAIN}/wav.scp,speech,sound "
        f"--train_data_path_and_name_and_type {mp}/{TRAIN}/text,text,text "
        f"--valid_data_path_and_name_and_type {mp}/{VALID}/wav.scp,speech,sound "
        f"--valid_data_path_and_name_and_type {mp}/{VALID}/text,text,text "
        f"--ngpu {num_gpus} "
        f"--pretrain_key {None} "
        f"--pretrain_path {pretrain_config['pretrained_model_file']} " #TODO(tilo)
        f"--multiprocessing_distributed {is_distributed} "
    )
    if not collect_stats:
        argString += (
            f"--train_shape_file {sp}/train/speech_shape "
            f"--train_shape_file {sp}/train/text_shape "
            f"--valid_shape_file {sp}/valid/speech_shape "
            f"--valid_shape_file {sp}/valid/text_shape "
        )
    else:
        argString += (
            f"--train_shape_file {mp}/{TRAIN}/wav.scp "
            f"--valid_shape_file {mp}/{VALID}/wav.scp "
        )
    parser = ASRTask.get_parser()
    args = parser.parse_args(shlex.split(argString))
    if pretrain_config is not None:
        args.encoder_conf = pretrain_config["encoder_conf"]
        args.decoder_conf = pretrain_config["decoder_conf"]
        args.normalize = pretrain_config["normalize"]
        d = pretrain_config["normalize_conf"]
        d["stats_file"]=f"{pretrain_config['pretrained_base']}/exp/asr_stats_raw_sp/train/feats_stats.npz"# TODO(tilo)
        args.normalize_conf = d

    ASRTask.main(args=args)


CONFIG_YML = "config.yml"


def run_espnet(args: argparse.Namespace):
    out_path = args.output_path
    os.makedirs(out_path, exist_ok=True)
    config_file = f"{out_path}/{CONFIG_YML}"
    if args.config_yml is None:
        data_io.write_file(config_file, build_config(args))
    else:
        shutil.copyfile(args.config_yml, config_file)

    build_manifest_files(
        f"{out_path}/{MANIFESTS}/{TRAIN}", args.train_path, limit=args.train_limit
    )
    build_manifest_files(
        f"{out_path}/{MANIFESTS}/{VALID}", args.eval_path, limit=args.eval_limit
    )

    if not os.path.isdir(f"{out_path}/{TOKENIZER}"):
        train_tokenizer(out_path, args.vocab_size)

    if not os.path.isdir(f"{out_path}/{STATS}"):
        run_asr_task(
            out_path,
            config_file,
            bpe_model=f"{out_path}/{TOKENIZER}/bpe.model",
            token_list=f"{out_path}/{TOKENIZER}/tokens.txt",
            collect_stats=True,
        )

    run_asr_task(
        out_path,
        config_file,
        num_workers=args.num_workers,
        bpe_model=f"{out_path}/{TOKENIZER}/bpe.model",
        token_list=f"{out_path}/{TOKENIZER}/tokens.txt",
        num_gpus=args.num_gpus,
        is_distributed=args.is_distributed,
    )


def finetune_espnet(args: argparse.Namespace):
    pretrained_base = args.pretrained_base
    meta_yml = f"{pretrained_base}/meta.yaml"
    with open(meta_yml, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    config_yml = (
        f"{pretrained_base}/{meta['yaml_files']['asr_train_config']}"
    )
    bpe_model = f"{pretrained_base}/data/token_list/bpe_unigram5000/bpe.model"
    with open(config_yml, "r", encoding="utf-8") as f:
        pretrain_config = yaml.safe_load(f)

    pretrain_config["pretrained_base"]=pretrained_base
    pretrain_config["pretrained_model_file"]=f"{pretrained_base}/{meta['files']['asr_model_file']}"

    out_path = args.output_path
    os.makedirs(out_path, exist_ok=True)
    config_file = f"{out_path}/{CONFIG_YML}"
    if args.config_yml is None:
        data_io.write_file(config_file, build_config(args))
    else:
        shutil.copyfile(args.config_yml, config_file)

    build_manifest_files(
        f"{out_path}/{MANIFESTS}/{TRAIN}", args.train_path, limit=args.train_limit
    )
    build_manifest_files(
        f"{out_path}/{MANIFESTS}/{VALID}", args.eval_path, limit=args.eval_limit
    )
    data_io.write_lines("tokens.txt", pretrain_config["token_list"])
    if not os.path.isdir(f"{out_path}/{STATS}") or True:
        run_asr_task(
            out_path,
            config_file,
            bpe_model=bpe_model,
            token_list="tokens.txt",
            pretrain_config=pretrain_config,
            collect_stats=True,
        )

    run_asr_task(
        out_path,
        config_file,
        num_workers=args.num_workers,
        bpe_model=bpe_model,
        token_list="tokens.txt",
        num_gpus=args.num_gpus,
        pretrain_config=pretrain_config,
        is_distributed=args.is_distributed,
    )


os.environ["LRU_CACHE_CAPACITY"] = str(1)
# see [Memory leak when evaluating model on CPU with dynamic size tensor input](https://github.com/pytorch/pytorch/issues/29893) and [here](https://raberrytv.wordpress.com/2020/03/25/pytorch-free-your-memory/)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument('--num_gpus', type=int, default=0) # used to support multi-GPU or CPU training
    parser.add_argument('--train_path', type=str, default=os.environ["HOME"]+"/data/asr_data/ENGLISH/LibriSpeech/dev-clean_preprocessed")
    parser.add_argument('--eval_path', type=str, default=os.environ["HOME"]+"/data/asr_data/ENGLISH/LibriSpeech/dev-clean_preprocessed")
    parser.add_argument('--output_path', type=str, default="/tmp/espnet_output")
    parser.add_argument('--is_distributed', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_encoder_blocks', type=int, default=1)
    parser.add_argument('--train_limit', type=int, default=1)
    parser.add_argument('--eval_limit', type=int, default=None)
    parser.add_argument('--config_yml', type=str, default=None)
    parser.add_argument('--pretrained_base', type=str, default=os.environ["HOME"]+"/data/espnet_pretrained")
    parser.add_argument('--vocab_size', type=int, default=500)
    # fmt:on
    args = parser.parse_args()
    print(args)

    finetune_espnet(args)

    """
    LRU_CACHE_CAPACITY=1 python ~/code/SPEECH/espnet/espnet2/bin/main.py

    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:26,464 (trainer:243) INFO: 1epoch results: [train] iter_time=0.164, forward_time=0.952, loss=480.386, loss_att=191.654, loss_ctc=1.154e+03, acc=1.022e-04, backward_time=0.973, optim_step_time=0.007, lr_0=2.080e-06, train_time=2.105, time=1 minute and 43.19 seconds, total_count=49, [valid] loss=479.955, loss_att=191.637, loss_ctc=1.153e+03, acc=1.030e-04, cer=1.010, wer=1.000, cer_ctc=4.993, time=50.5 seconds, total_count=49, [att_plot] time=3.2 seconds, total_count=0
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,067 (trainer:286) INFO: The best model has been updated: valid.acc
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,068 (trainer:322) INFO: The training was finished at 1 epochs 

    """
