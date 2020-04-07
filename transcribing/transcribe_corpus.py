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
from utils import USE_GPU, BLANK_SYMBOL, SPACE
from asr_checkpoint import load_evaluatable_checkpoint

warnings.simplefilter("ignore")


import torch


def run_transcribtion(
    samples: Iterable[Sample], model_path, use_beam_decoder=False, use_half=False
):

    device = torch.device("cuda" if USE_GPU else "cpu")
    model,data_conf,audio_conf = load_evaluatable_checkpoint(device, model_path, use_half)

    char2idx = dict([(data_conf.labels[i], i) for i in range(len(data_conf.labels))])
    decoder = build_decoder(char2idx, use_beam_decoder)

    audio_fe = AudioFeatureExtractor(audio_conf, [])

    def do_transcribe(audio_file):
        decoded_output, decoded_offsets = prediction(
            audio_path=audio_file,
            fe=audio_fe,
            model=model,
            decoder=decoder,
            device=device,
            use_half=use_half,
        )
        candidate_idx = 0
        return [x[candidate_idx] for x in decoded_output][0]

    for s in tqdm(samples):
        prediction = do_transcribe(s.audio_file)
        target = s.text
        yield prediction, target


if __name__ == "__main__":
    HOME = os.environ["HOME"]
    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"
    samples = build_librispeech_corpus(
        raw_data_path, "eval", ["dev-clean", "dev-other"]
    )

    transcribtions = run_transcribtion(samples[:100], "/tmp/deepspeech_9.pth.tar")
    data_io.write_jsonl(
        "/tmp/predictions.jsonl",
        ({"prediction": p, "target": t} for p, t in transcribtions),
    )
