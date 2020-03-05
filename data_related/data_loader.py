import os
import subprocess
from tempfile import NamedTemporaryFile

import torchaudio
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler

import librosa
import numpy as np
import scipy.signal
import torch
from scipy.io.wavfile import read
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# from data_related.data_augmentation.spec_augment import spec_augment
from data_related.data_augmentation.signal_augment import random_augmentation

windows = {
    "hamming": scipy.signal.hamming,
    "hann": scipy.signal.hann,
    "blackman": scipy.signal.blackman,
    "bartlett": scipy.signal.bartlett,
}

SAMPLE_RATE = 16_000


def load_audio(path):
    sample_rate, sound = read(path)
    sound = sound.astype("float32") / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


def get_feature_dim(audio_conf):
    feature_type = (
        audio_conf["feature_type"] if "feature_type" in audio_conf else "stft"
    )
    if feature_type == "mfcc":
        FEATURE_DIM = 40
    elif feature_type == "mel":
        FEATURE_DIM = 161
    elif feature_type == "stft":
        FEATURE_DIM = int(
            math.floor((audio_conf["sample_rate"] * audio_conf["window_size"]) / 2) + 1
        )  # 161
    else:
        assert False
    return FEATURE_DIM


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


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    def __init__(
        self,
        audio_conf,
        normalize=False,
        signal_augment=False,
        spec_augment=False,
    ):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        super(SpectrogramParser, self).__init__()
        self.feature_type = audio_conf.get("feature_type", "stft")
        self.window_stride = audio_conf["window_stride"]
        self.window_size = audio_conf["window_size"]
        self.sample_rate = audio_conf["sample_rate"]
        self.window = windows.get(audio_conf["window"], windows["hamming"])
        self.normalize = normalize
        self.signal_augment = signal_augment
        self.spec_augment = spec_augment

        if self.feature_type == "mfcc":
            self.mfcc = torchaudio.transforms.MFCC(
                sample_rate=SAMPLE_RATE, n_mfcc=get_feature_dim(audio_conf)
            )
        elif self.feature_type == "mel":
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_mels=get_feature_dim(audio_conf)
            )

        self.audio_files = [
            f for f, _ in self.audio_text_files
        ]  # TODO accessing the child-classes attribute!!

    def parse_audio(self, audio_path):
        if self.signal_augment:
            y = load_randomly_augmented_audio(audio_path, self.audio_files)
        else:
            y = load_audio(audio_path)

        if self.feature_type == "mfcc":
            feat = self.mfcc.forward(torch.from_numpy(y).unsqueeze(0)).data.squeeze(0)
        elif self.feature_type == "stft":
            feat = self._calc_stft(y)
        elif self.feature_type == "mel":
            feat = self.mel.forward(torch.from_numpy(y).unsqueeze(0)).data.squeeze(0)
        else:
            assert False

        if self.normalize:
            mean = feat.mean()
            std = feat.std()
            feat.add_(-mean)
            feat.div_(std)

        return feat

    def _calc_stft(self, y):
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
        )
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)

        if self.spec_augment:
            spect = spec_augment(spect)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(
        self,
        audio_conf,
        manifest_filepath,
        labels,
        normalize=False,
        signal_augment=False,
        spec_augment=False,
    ):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        with open(manifest_filepath) as f:
            audio_text_files = f.readlines()

        if (
            "libri" in manifest_filepath
        ):  # TODO(tilo): just too lazy to rebuild the csvs

            def fix_path(wav_file, txt_file):
                if "libri_train100_manifest.csv" == manifest_filepath:
                    base_path = "LibriSpeech_train_100"
                else:
                    base_path = "LibriSpeech_dataset"

                def splitit(s):
                    tmp = s.split(base_path)
                    if len(tmp) != 2:
                        print(s)
                        assert False
                    _, f = tmp
                    return f

                wav_file = "/".join([base_path, splitit(wav_file)])
                txt_file = "/".join([base_path, splitit(txt_file)])
                return wav_file, txt_file

            audio_text_files = [
                fix_path(*x.strip().split(",")) for x in audio_text_files
            ]
        else:
            audio_text_files = [x.strip().split(",") for x in audio_text_files]

        self.audio_text_files = audio_text_files
        self.size = len(audio_text_files)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(
            audio_conf, normalize, signal_augment, spec_augment
        )

    def __getitem__(self, index):
        audio_file, text_file = self.audio_text_files[index]
        spect = self.parse_audio(audio_file)
        transcript = self.parse_transcript(text_file)
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, "r", encoding="utf8") as f:
            transcript = f.read().replace("\n", "")
        transcript = list(
            filter(None, [self.labels_map.get(x) for x in list(transcript)])
        )  # TODO(tilo) like this it erases unknown letters
        # transcript = [self.labels_map.get(x,UNK) for x in list(transcript)] # TODO(tilo) better like this?
        return transcript

    def __len__(self):
        return self.size


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


def get_audio_length(path):
    output = subprocess.check_output(['soxi -D "%s"' % path.strip()], shell=True)
    return float(output)


def load_randomly_augmented_audio(path, audio_files):
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        while True:
            try:
                random_augmentation(path, audio_files, augmented_filename)
                audio = load_audio(augmented_filename)
                break
            except:
                pass

    return audio


if __name__ == '__main__':
    x = load_randomly_augmented_audio('/tmp/original.wav',['/tmp/interfere.wav','/tmp/interfere2.wav'])