import argparse
import warnings
from data_related.audio_feature_extraction import (
    AudioFeatureExtractor,
    AudioFeaturesConfig,
)
from utils import load_model, BLANK_SYMBOL, SPACE, HOME

warnings.simplefilter("ignore")

from decoder import GreedyDecoder, DecoderConfig

import torch


def transcribe(audio_path, fe: AudioFeatureExtractor, model, decoder, device, use_half):
    spect = fe.process(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        print("using half")
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets


def build_decoder(char2idx, use_beam_decoder=False, config=DecoderConfig()):
    if use_beam_decoder:
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(char2idx, **config._asdict())
    else:
        decoder = GreedyDecoder(char2idx)
    return decoder


if __name__ == "__main__":
    use_half = False
    use_beam_decoder = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = "/tmp/deepspeech_9.pth.tar"
    audio_file = (
        HOME
        + "/data/asr_data/ENGLISH/LibriSpeech/dev-other/8288/274162/8288-274162-0016.flac"
    )

    model = load_model(device, model_file, use_half)

    # fmt: off
    labels = [BLANK_SYMBOL, "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", SPACE]
    # fmt: on
    char2idx = dict([(labels[i], i) for i in range(len(labels))])

    decoder = build_decoder(char2idx,use_beam_decoder)

    audio_conf = AudioFeaturesConfig()
    audio_fe = AudioFeatureExtractor(audio_conf, [])

    decoded_output, decoded_offsets = transcribe(
        audio_path=audio_file,
        fe=audio_fe,
        model=model,
        decoder=decoder,
        device=device,
        use_half=use_half,
    )
    print(decoded_output[0][0].encode("utf-8"))
    with open("output.txt", "wb") as f:
        f.write(decoded_output[0][0].encode("utf-8"))
    # print()

    # 'siseñora, ceramucho  de cuarita untaas , rectesart  nombre blan carian ayadías.'
    # ['del pío del la de poretuio togamos, total porque e primero porque yo era una menificiaria del rezo gamoso, e mí.']
