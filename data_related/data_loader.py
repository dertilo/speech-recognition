from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

import numpy as np
import torch
import math
from tqdm import tqdm



# def load_audio(path):
#     sound, sample_rate = torchaudio.load(path, normalization=True)
#     if sample_rate != SAMPLE_RATE:
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,new_freq=SAMPLE_RATE)
#         sound = resampler(sound)
#     sound = sound.numpy().T
#     if len(sound.shape) > 1:
#         if sound.shape[1] == 1:
#             sound = sound.squeeze()
#         else:
#             sound = sound.mean(axis=1)  # multiple channels, average
#     return sound


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)  # TODO(tilo): why should one shuffle here?
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [
            self.ids[i : i + batch_size] for i in range(0, len(self.ids), batch_size)
        ]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[: (self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[
            offset :: self.num_replicas
        ]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]


if __name__ == "__main__":
    audio_conf = dict(
        sample_rate=16_000,
        window_size=0.02,
        window_stride=0.01,
        window="hamming",
        feature_type="stft",
    )
    # fmt: off
    labels = ["_", "'","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," "]
    # fmt: on

    train_dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath='libri_val_manifest.csv',
        labels=labels,
        normalize=True,
        signal_augment=False,
        spec_augment=False,
    )

    train_sampler = BucketingSampler(train_dataset, batch_size=32)

    train_loader = AudioDataLoader(
        train_dataset, num_workers=0, batch_sampler=train_sampler
    )

    for d in tqdm(train_loader):
        pass

    # x = load_randomly_augmented_audio(
    #     "/tmp/original.wav", ["/tmp/interfere.wav", "/tmp/interfere2.wav"]
    # )
