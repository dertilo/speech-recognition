import argparse
import os

import torch
from tqdm import tqdm
from util import data_io

from asr_checkpoint import load_evaluatable_checkpoint
from data_related.char_stt_dataset import CharSTTDataset
from data_related.data_loader import AudioDataLoader
from data_related.librispeech import build_librispeech_corpus
from decoder import GreedyDecoder
from transcribing.transcribe_util import build_decoder, transcribe_batch
from utils import (
    HOME,
    USE_GPU,
)


def run_transcription(
    test_loader,
    device,
    model,
    decoder,
    target_decoder,
    half=False,
):
    model.eval()
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        decoded_output, _, _ = transcribe_batch(
            decoder, device, half, input_percentages, inputs, model
        )

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset : offset + size])
            offset += size

        target_strings = target_decoder.convert_to_strings(split_targets)
        yield [x[0] for x in decoded_output],[x[0] for x in target_strings]


# fmt: off
parser = argparse.ArgumentParser(description="args")
parser.add_argument("--model", type=str,default='deepspeech_9.pth.tar')
parser.add_argument("--datasets", type=str,nargs='+', default=['test-clean'])
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--out-dir", type=str, default='transcriptions')
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

    decoder = build_decoder(char2idx, use_beam_decoder=False)

    target_decoder = GreedyDecoder(char2idx)

    asr_path = HOME + "/data/asr_data"
    out_path = os.path.join(asr_path, args.out_dir)
    os.makedirs(out_path, exist_ok=True)
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"
    samples = build_librispeech_corpus(
        raw_data_path, "_".join(args.datasets), args.datasets
    )
    samples = samples

    dataset = CharSTTDataset(samples, conf=data_conf, audio_conf=audio_conf, )
    test_loader = AudioDataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    g = run_transcription(
        test_loader=test_loader,
        device=device,
        model=model,
        decoder=decoder,
        target_decoder=target_decoder,
        half=use_half,
    )
    # i = iter(g)
    # batches = [next(i) for _ in range(5)]
    batches = list(g)
    data_io.write_lines(os.path.join(out_path,'hypos.txt'), (h for hs,ts in batches for h in hs))
    data_io.write_lines(os.path.join(out_path,'targets.txt'), (t for hs,ts in batches for t in ts))