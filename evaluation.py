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
from metrics_calculation import calc_num_word_errors, calc_num_char_erros
from transcribing.transcribe import build_decoder
from utils import (
    reduce_tensor,
    calc_loss,
    BLANK_SYMBOL,
    SPACE,
    HOME,
    USE_GPU,
)
from asr_checkpoint import load_evaluatable_checkpoint


def transcribe_batch(decoder, device, half, input_percentages, inputs, model):
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    inputs = inputs.to(device)
    if half:
        inputs = inputs.half()
    out, output_sizes = model(inputs, input_sizes)
    probs = F.softmax(out, dim=-1)
    decoded_output, _ = decoder.decode(probs, output_sizes)
    return decoded_output, out, output_sizes


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
        decoded_output, out, output_sizes = transcribe_batch(
            decoder, device, half, input_percentages, inputs, model
        )

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

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset : offset + size])
            offset += size

        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append(
                (out.cpu().numpy(), output_sizes.numpy(), target_strings)
            )
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = calc_num_word_errors(transcript, reference)
            cer_inst = calc_num_char_erros(transcript, reference)
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


# fmt: off
parser = argparse.ArgumentParser(description="args")
parser.add_argument("--model", type=str,default='deepspeech_9.pth.tar')
parser.add_argument("--datasets", type=str,nargs='+', default='test-clean')
# fmt: on

if __name__ == "__main__":
    """
    python evaluation.py --model libri_960_1024_32/deepspeech_8.pth.tar --datasets test-clean
    """
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if USE_GPU else "cpu")
    use_half = False
    model, data_conf, audio_conf = load_evaluatable_checkpoint(
        device, HOME + "/data/asr_data/checkpoints/%s" % args.model, use_half
    )

    char2idx = dict([(data_conf.labels[i], i) for i in range(len(data_conf.labels))])

    decoder = build_decoder(char2idx, use_beam_decoder=True)

    target_decoder = GreedyDecoder(char2idx)

    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"
    samples = build_librispeech_corpus(
        raw_data_path, "_".join(args.datasets), args.datasets
    )
    samples = samples

    test_dataset = CharSTTDataset(samples, conf=data_conf, audio_conf=audio_conf,)
    test_loader = AudioDataLoader(test_dataset, batch_size=20, num_workers=4)
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
