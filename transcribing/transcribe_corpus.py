import argparse
import gzip
import os
import warnings

from tqdm import tqdm
from typing import Iterable
from util import data_io

from data_related.audio_feature_extraction import (
    AudioFeaturesConfig,
    AudioFeatureExtractor,
)
from data_related.char_stt_dataset import CharSTTDataset, DataConfig
from data_related.audio_util import Sample
from data_related.librispeech import build_librispeech_corpus
from model import DeepSpeech
from transcribing.transcribe import transcribe, build_decoder
from utils import load_model, USE_GPU, BLANK_SYMBOL, SPACE

warnings.simplefilter("ignore")


import torch

def run_transcribtion(
    samples: Iterable[Sample], model_path, use_beam_decoder=False, use_half=False
):

    device = torch.device("cuda" if USE_GPU else "cpu")
    model: DeepSpeech = load_model(device, model_path, use_half)

    # fmt: off
    labels = [BLANK_SYMBOL, "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", SPACE]
    # fmt: on
    char2idx = dict([(labels[i], i) for i in range(len(labels))])
    decoder = build_decoder(char2idx, use_beam_decoder)

    audio_conf = AudioFeaturesConfig()
    audio_fe = AudioFeatureExtractor(audio_conf, [])

    def do_transcribe(audio_file):
        decoded_output, decoded_offsets = transcribe(
            audio_path=audio_file,
            fe=audio_fe,
            model=model,
            decoder=decoder,
            device=device,
            use_half=use_half,
        )
        candidate_idx = 0
        return [x[candidate_idx] for x in decoded_output][0]

    g = ((do_transcribe(s.audio_file), s.text) for s in tqdm(samples))
    return list(g)


if __name__ == "__main__":
    HOME = os.environ["HOME"]
    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"
    samples = build_librispeech_corpus(
        raw_data_path, "eval", ["dev-clean", "dev-other"]
    )

    transcribed = [
        (a, b)
        for a, b in [("---", "---")]
        + run_transcribtion(samples[:100], "/tmp/deepspeech_9.pth.tar")
    ]

    lines = (
        ["| " + " | ".join(["prediction", "target"])]
        + ["| " + " | ".join(["---", "---"])]
        + ["| " + " | ".join([a, b]) for a, b in transcribed]
    )
    data_io.write_lines("/tmp/test.txt", lines)
