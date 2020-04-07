import argparse
import torch.nn.functional as F

import numpy as np
import torch
from tqdm import tqdm

from data_related.audio_feature_extraction import AudioFeaturesConfig
from data_related.char_stt_dataset import CharSTTDataset, DataConfig
from data_related.data_loader import AudioDataLoader
from data_related.librispeech import build_librispeech_corpus
from decoder import GreedyDecoder
from metrics_calculation import calc_wer, calc_cer
from transcribing.transcribe import build_decoder
from utils import (
    load_model,
    reduce_tensor,
    calc_loss,
    BLANK_SYMBOL,
    SPACE,
    HOME,
    USE_GPU,
)


def evaluate(
    test_loader,
    device,
    model,
    decoder,
    target_decoder,
    criterion=None,
    args=None,
    save_output=False,
    verbose=False,
    half=False,
):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    avg_loss = 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        if half:
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset : offset + size])
            offset += size
        out, output_sizes = model(inputs, input_sizes)
        if criterion is not None:
            _, loss_value = calc_loss(
                out,
                output_sizes,
                criterion,
                targets,
                target_sizes,
                device,
                args.distributed,
                args.world_size,
            )
            avg_loss += loss_value

        probs = F.softmax(out, dim=-1)
        decoded_output, _ = decoder.decode(probs, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append(
                (out.cpu().numpy(), output_sizes.numpy(), target_strings)
            )
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = calc_wer(transcript, reference)
            cer_inst = calc_cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(" ", ""))
            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print(
                    "WER:",
                    float(wer_inst) / len(reference.split()),
                    "CER:",
                    float(cer_inst) / len(reference.replace(" ", "")),
                    "\n",
                )
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    avg_loss /= i
    # print("avg valid loss %0.2f" % avg_loss)
    return wer * 100, cer * 100, avg_loss, output_data


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if USE_GPU else "cpu")
    use_half = False
    model = load_model(device, "/tmp/deepspeech_9.pth.tar", use_half)

    # fmt: off
    labels = [BLANK_SYMBOL, "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", SPACE]
    # fmt: on
    char2idx = dict([(labels[i], i) for i in range(len(labels))])
    conf = DataConfig(labels)

    decoder = build_decoder(char2idx)

    target_decoder = GreedyDecoder(char2idx)

    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"
    samples = build_librispeech_corpus(raw_data_path, "eval", ["dev-clean"])
    samples = samples[:100]
    audio_conf = AudioFeaturesConfig()

    test_dataset = CharSTTDataset(samples, conf=conf, audio_conf=audio_conf,)
    test_loader = AudioDataLoader(test_dataset, batch_size=16, num_workers=0)
    wer, cer, avg_loss, output_data = evaluate(
        test_loader=test_loader,
        device=device,
        model=model,
        decoder=decoder,
        target_decoder=target_decoder,
        save_output=False,
        verbose=False,
        half=use_half,
    )

    print(
        "Test Summary \t"
        "Average WER {wer:.3f}\t"
        "Average CER {cer:.3f}\t".format(wer=wer, cer=cer)
    )
