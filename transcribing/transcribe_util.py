import argparse
import warnings
from data_related.audio_feature_extraction import (
    AudioFeatureExtractor,
    AudioFeaturesConfig,
)
from utils import BLANK_SYMBOL, SPACE, HOME

warnings.simplefilter("ignore")
import torch.nn.functional as F
from decoder import GreedyDecoder, DecoderConfig, Decoder

import torch

def transcribe_batch(decoder:Decoder, device, half:bool, input_len_proportions, inputs, model):
    input_sizes = input_len_proportions.mul_(int(inputs.size(3))).int()
    inputs = inputs.to(device)
    if half:
        inputs = inputs.half()
    out, output_sizes = model(inputs, input_sizes)
    probs = F.softmax(out, dim=-1)
    decoded_output, _ = decoder.decode(probs, output_sizes)
    return decoded_output, out, output_sizes

def transcribe_single(
    audio_path, fe: AudioFeatureExtractor, model, decoder, device, use_half
):
    spect = fe.process(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        print("using half")
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    probs = F.softmax(out, dim=-1)
    decoded_output, decoded_offsets = decoder.decode(probs, output_sizes)
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

    model, data_conf, audio_conf = load_evaluatable_checkpoint( #TODO(tilo)
        device, model_file, use_half
    )

    char2idx = dict([(data_conf.labels[i], i) for i in range(len(data_conf.labels))])

    decoder = build_decoder(char2idx, use_beam_decoder)

    audio_fe = AudioFeatureExtractor(audio_conf, [])

    decoded_output, decoded_offsets = transcribe_single(
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
