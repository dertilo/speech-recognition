import argparse
import gzip
import warnings

from tqdm import tqdm

from data_related.data_utils import write_lines
from data_related.vocabulary import BLANK_CHAR
from model import DeepSpeech
from opts import add_decoder_args, add_inference_args
from transcribing.transcribe import transcribe
from utils import load_model

warnings.simplefilter("ignore")

from decoder import GreedyDecoder


import torch

from data_related.data_loader import SpectrogramParser


def read_lines(file, mode="b", encoding="utf-8", limit=None):
    assert any([mode == m for m in ["b", "t"]])
    counter = 0
    with gzip.open(file, mode="r" + mode) if file.endswith(".gz") else open(
        file, mode="r" + mode
    ) as f:
        for line in f:
            counter += 1
            if limit and (counter > limit):
                break
            if "b" in mode:
                line = line.decode(encoding)
            yield line.replace("\n", "")


def run_transcribtion(args, model_path):
    device = torch.device("cuda" if args.cuda else "cpu")
    model: DeepSpeech = load_model(device, model_path, args.half)
    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(
            model.labels,
            lm_path=args.lm_path,
            alpha=args.alpha,
            beta=args.beta,
            cutoff_top_n=args.cutoff_top_n,
            cutoff_prob=args.cutoff_prob,
            beam_width=args.beam_width,
            num_processes=args.lm_workers,
        )
    else:
        decoder = GreedyDecoder(
            model.labels, blank_index=model.labels.index(BLANK_CHAR)
        )
    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

    def do_transcribe(audio_file):
        decoded_output, decoded_offsets = transcribe(
            audio_path=audio_file,
            spect_parser=spect_parser,
            model=model,
            decoder=decoder,
            device=device,
            use_half=args.half,
        )
        candidate_idx = 0
        return [x[candidate_idx] for x in decoded_output][0]

    def fix_path(l):
        if "libri" in l:
            path = l.replace(
                "/beegfs/home/users/t/tilo-himmelsbach/SPEECH/deepspeech.pytorch/", ""
            )
        else:
            path = l
        return path

    lines_g = read_lines(args.manifest)
    [next(lines_g) for _ in range(2_000)]
    lines = [next(lines_g) for _ in range(10)]
    examples_g = (fix_path(l).split(",") for l in lines)
    g = (
        (do_transcribe(audio_file), next(iter(read_lines(text_file))))
        for audio_file, text_file in tqdm(examples_g)
    )
    # writer = MarkdownTableWriter()
    # writer.table_name = "write example with a margin"
    # writer.headers = ["int", "float", "str", "bool", "mix", "time"]
    # writer.value_matrix = [
    #     [0,   0.1,      "hoge", True,   0,      "2017-01-01 03:04:05+0900"],
    #     [2,   "-2.23",  "foo",  False,  None,   "2017-12-23 45:01:23+0900"],
    #     [3,   0,        "bar",  "true",  "inf", "2017-03-03 33:44:55+0900"],
    #     [-10, -9.9,     "",     "FALSE", "nan", "2017-01-01 00:00:00+0900"],
    # ]
    # writer.margin = 1  # add a whitespace for both sides of each cell
    #
    # writer.write_table()
    return list(g)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="DeepSpeech transcription")
    arg_parser = add_inference_args(arg_parser)
    arg_parser.add_argument("--manifest")
    arg_parser.add_argument("--output", default="transcribed.csv")
    arg_parser.add_argument(
        "--offsets",
        dest="offsets",
        action="store_true",
        help="Returns time offset information",
    )
    arg_parser = add_decoder_args(arg_parser)
    args = arg_parser.parse_args()

    # model_path = args.model_path

    transcribed = [
        (a, b)
        for k in [40, 2]
        for a, b in [("---", "---")]
        + run_transcribtion(
            args, "checkpoints/spanish_augmented/deepspeech_%d.pth.tar" % k
        )
    ]

    lines = (
        ["| " + " | ".join(["prediction", "target"])]
        + ["| " + " | ".join(["---", "---"])]
        + ["| " + " | ".join([a, b]) for a, b in transcribed]
    )
    write_lines(
        args.output, lines,
    )
