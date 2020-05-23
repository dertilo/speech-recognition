import argparse

import torch
from tqdm import tqdm

from data_related.audio_feature_extraction import AudioFeaturesConfig
from data_related.char_stt_dataset import CharSTTDataset, DataConfig
from data_related.data_loader import AudioDataLoader
from data_related.librispeech import build_librispeech_corpus, LIBRI_VOCAB
from decoder import GreedyDecoder
from metrics_calculation import calc_num_word_errors, calc_num_char_erros
from model import DeepSpeech
from transcribing.transcribe_util import build_decoder, transcribe_batch
from utils import (
    calc_loss,
    HOME,
    USE_GPU, unflatten_targets,
)
from asr_checkpoint import load_evaluatable_checkpoint


def calc_loss_value(args, criterion, device, out, output_sizes, target_sizes, targets):
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
    else:
        loss_value = 0
    return loss_value


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
    calc_loss_value_fun=calc_loss_value,
):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    avg_loss = 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data

        (
            loss_value,
            num_chars_step,
            num_tokens_step,
            total_cer_step,
            total_wer_step,
        ) = validation_step(
            args,
            calc_loss_value_fun,
            criterion,
            decoder,
            device,
            half,
            input_percentages,
            inputs,
            model,
            output_data,
            save_output,
            target_decoder,
            target_sizes,
            targets,
            verbose,
        )
        avg_loss += loss_value
        num_chars += num_chars_step
        num_tokens += num_tokens_step
        total_cer += total_cer_step
        total_wer += total_wer_step

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    avg_loss /= i
    # print("avg valid loss %0.2f" % avg_loss)
    return wer * 100, cer * 100, avg_loss, output_data


def validation_step(
    args,
    calc_loss_value_fun,
    criterion,
    decoder,
    device,
    half,
    input_percentages,
    inputs,
    model,
    output_data,
    save_output,
    target_decoder,
    target_sizes,
    targets,
    verbose,
):
    decoded_output, out, output_sizes = transcribe_batch(
        decoder, device, half, input_percentages, inputs, model
    )
    loss_value = calc_loss_value_fun(
        args, criterion, device, out, output_sizes, target_sizes, targets
    )
    (num_chars_step, num_tokens_step, total_cer_step, total_wer_step,) = calc_errors(
        decoded_output,
        out,
        output_data,
        output_sizes,
        save_output,
        target_decoder,
        target_sizes,
        targets,
        verbose,
    )
    return loss_value, num_chars_step, num_tokens_step, total_cer_step, total_wer_step


def calc_errors(
    decoded_output,
    out,
    output_data,
    output_sizes,
    save_output,
    target_decoder,
    target_sizes,
    targets,
    verbose,
):
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    split_targets = unflatten_targets(targets, target_sizes)
    target_strings = target_decoder.convert_to_strings(split_targets)
    if save_output is not None:
        # add output to data array, and continue
        output_data.append((out.cpu().numpy(), output_sizes.numpy(), target_strings))
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
    return num_chars, num_tokens, total_cer, total_wer


# fmt: off
parser = argparse.ArgumentParser(description="args")
parser.add_argument("--model", type=str,default='libri_960_1024_32_11_04_2020/deepspeech_9.pth.tar')
parser.add_argument("--datasets", type=str,nargs='+', default='test-clean')
# fmt: on

if __name__ == "__main__":
    """
    python evaluation.py --model libri_960_1024_32_11_04_2020/deepspeech_9.pth.tar --datasets test-clean
    :returns 
    BeamCTCDecoder: Test Summary    Average WER 8.936       Average CER 2.962
    GreedyDecoder: Test Summary    Average WER 9.059       Average CER 2.998
    """
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if USE_GPU else "cpu")
    use_half = False
    checkpoint_file = HOME + "/data/asr_data/checkpoints/%s" % args.model

    if "lit" in checkpoint_file:
        from lightning.lightning_model import LitSTTModel
        def load_model_from_lightning_checkpoint(file):
            model: DeepSpeech = LitSTTModel.load_from_checkpoint(file).model.eval()
            data_conf = DataConfig(LIBRI_VOCAB)
            audio_conf = AudioFeaturesConfig()
            return model, data_conf, audio_conf


        model, data_conf, audio_conf = load_model_from_lightning_checkpoint(
            checkpoint_file
        )
        model = model.to(device)
    else:
        model, data_conf, audio_conf = load_evaluatable_checkpoint(
            device, checkpoint_file, use_half
        )

    char2idx = dict([(data_conf.labels[i], i) for i in range(len(data_conf.labels))])

    decoder = build_decoder(char2idx, use_beam_decoder=False)

    target_decoder = GreedyDecoder(char2idx)

    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"
    samples = build_librispeech_corpus(
        raw_data_path, "_".join(args.datasets), args.datasets
    )

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
