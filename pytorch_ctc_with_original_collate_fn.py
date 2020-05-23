import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from data_related.data_loader import AudioDataLoader, BucketingSampler
from data_related.librispeech import build_dataset
from model import DeepSpeech
from utils import unflatten_targets, calc_loss, BLANK_SYMBOL, USE_GPU


def build_ctc_loss(blank,device):
    def loss_fun(float_out: torch.Tensor, targets, output_sizes, target_sizes):
        targets = unflatten_targets(targets, target_sizes)
        targets = [torch.IntTensor(target).to(device) for target in targets]
        padded_target = pad_sequence(targets, batch_first=True)

        prob = F.log_softmax(float_out, -1)
        ctc_loss = F.ctc_loss(
            prob,
            padded_target,
            output_sizes,
            target_sizes,
            blank=blank,
            zero_infinity=True,
        )
        return ctc_loss

    return loss_fun

device = torch.device("cuda" if USE_GPU else "cpu")

if __name__ == "__main__":
    ds = build_dataset("debug", ["dev-clean"])

    train_sampler = BucketingSampler(ds, batch_size=4)
    loader = AudioDataLoader(ds, num_workers=0, batch_sampler=train_sampler)
    blank = ds.char2idx[BLANK_SYMBOL]
    criterion = build_ctc_loss(blank=blank,device=device)

    model = DeepSpeech(
        hidden_size=64,
        nb_layers=2,
        vocab_size=len(ds.char2idx),
        input_feature_dim=ds.audio_fe.feature_dim,
        bidirectional=True,
    )

    for batch in loader:
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        # measure data loading time
        inputs = inputs.to(device)

        out, output_sizes = model(inputs, input_sizes)
        assert out.size(0) == inputs.size(0)
        loss, loss_value = calc_loss(
            out,
            output_sizes,
            criterion,
            targets,
            target_sizes,
            device,
            False,
            0,
        )
