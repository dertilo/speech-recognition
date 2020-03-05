import speech_recognition as sr
import argparse
import warnings

from opts import add_decoder_args, add_inference_args
from transcribing.transcribe import transcribe
from utils import load_model

warnings.simplefilter("ignore")

from decoder import GreedyDecoder

import torch

from data_related.data_loader import SpectrogramParser


def decode_results(decoded_output):
    pi = 0
    b = 0
    return decoded_output[b][pi]


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="DeepSpeech transcription")
    arg_parser = add_inference_args(arg_parser)

    arg_parser.add_argument(
        "--offsets",
        dest="offsets",
        action="store_true",
        help="Returns time offset information",
    )
    arg_parser = add_decoder_args(arg_parser)
    model_file = "/tmp/deepspeech_42.pth.tar"

    args = arg_parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, model_file, args.half)
    decoder = GreedyDecoder(model.labels, blank_index=model.labels.index("_"))
    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

    while True:
        r = sr.Recognizer()
        with sr.Microphone(sample_rate=16_000) as source:
            r.adjust_for_ambient_noise(source)
            print("listening ...")
            audio = r.listen(source)

        audio_file = "/tmp/test.wav"
        with open(audio_file, "wb") as f:
            bytes = audio.get_wav_data()
            f.write(bytes)

        print("recognizing ...")

        decoded_output, decoded_offsets = transcribe(
            audio_path=audio_file,
            spect_parser=spect_parser,
            model=model,
            decoder=decoder,
            device=device,
            use_half=args.half,
        )
        print(decode_results(decoded_output))
