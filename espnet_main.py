import os

import argparse

from espnet.espnet2.bin.tokenize_text import tokenize, get_parser
from espnet.espnet2.tasks.asr import ASRTask
from util import data_io, util_methods
import sentencepiece as spm
import shlex

TRAIN = "train"
VALID = "valid"
MANIFESTS = "manifests"
STATS = "stats"
TRAINLOGS = "train_logs"
TOKENIZER = "tokenizer"


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
        num_gpus=0,
        is_distributed=False,
        num_workers=0,
        collect_stats=False,
):
    sp = f"{output_path}/{STATS}"

    output_dir = sp if collect_stats else f"{output_path}/{TRAINLOGS}"
    mp = f"{output_path}/{MANIFESTS}"
    argString = (
        f"--collect_stats {collect_stats} "
        f"--use_preprocessor true "
        f"--bpemodel {output_path}/{TOKENIZER}/bpe.model "
        f"--seed 42 "
        f"--num_workers {num_workers} "
        f"--token_type bpe "
        f"--token_list {output_path}/{TOKENIZER}/tokens.txt "
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
    ASRTask.main(args=args)


def run_espnet(
        train_path,
        valid_path,
        out_path,
        vocab_size=500,
        limit=200,  # just for debug
        config="conf/tuning/train_asr_transformer_tiny.yaml",
        num_workers=0,
        num_gpus=0,
):
    build_manifest_files(f"{out_path}/{MANIFESTS}/{TRAIN}", train_path, limit=limit)
    build_manifest_files(f"{out_path}/{MANIFESTS}/{VALID}", valid_path, limit=limit)

    if not os.path.isdir(f"{out_path}/{TOKENIZER}"):
        train_tokenizer(out_path, vocab_size)

    if not os.path.isdir(f"{out_path}/{STATS}"):
        run_asr_task(out_path, config, collect_stats=True)

    run_asr_task(
        out_path,
        config,
        num_workers=num_workers,
        num_gpus=num_gpus,
        is_distributed=False,
    )


if __name__ == "__main__":
    os.environ["LRU_CACHE_CAPACITY"] = str(1)
    # see [Memory leak when evaluating model on CPU with dynamic size tensor input](https://github.com/pytorch/pytorch/issues/29893) and [here](https://raberrytv.wordpress.com/2020/03/25/pytorch-free-your-memory/)
    data_path = "/home/tilo/data/asr_data/ENGLISH/LibriSpeech/dev-clean-some_preprocessed"
    run_espnet(train_path=data_path, valid_path=data_path,
               out_path="/tmp/espnet_output")

    """
    LRU_CACHE_CAPACITY=1 python ~/code/SPEECH/espnet/espnet2/bin/main.py

    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:26,464 (trainer:243) INFO: 1epoch results: [train] iter_time=0.164, forward_time=0.952, loss=480.386, loss_att=191.654, loss_ctc=1.154e+03, acc=1.022e-04, backward_time=0.973, optim_step_time=0.007, lr_0=2.080e-06, train_time=2.105, time=1 minute and 43.19 seconds, total_count=49, [valid] loss=479.955, loss_att=191.637, loss_ctc=1.153e+03, acc=1.030e-04, cer=1.010, wer=1.000, cer_ctc=4.993, time=50.5 seconds, total_count=49, [att_plot] time=3.2 seconds, total_count=0
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,067 (trainer:286) INFO: The best model has been updated: valid.acc
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,068 (trainer:322) INFO: The training was finished at 1 epochs 


    """
