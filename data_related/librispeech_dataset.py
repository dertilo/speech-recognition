import math
import os
import subprocess
from tempfile import NamedTemporaryFile

import scipy.signal
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import NamedTuple, List
from util import data_io

from data_related.data_augmentation.signal_augment import random_augmentation
from data_related.data_augmentation.spec_augment import spec_augment
from data_related.data_utils import load_audio
from data_related.feature_extraction import calc_stft_librosa


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


NAME2WINDOWTYPE = {
    "hamming": scipy.signal.hamming,
    "hann": scipy.signal.hann,
    "blackman": scipy.signal.blackman,
    "bartlett": scipy.signal.bartlett,
}
SAMPLE_RATE = 16_000


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


class AudioFeaturesConfig(NamedTuple):
    sample_rate: int = 16_000
    window_size: float = 0.02
    window_stride: float = 0.01
    window: str = "hamming"
    feature_type: str = "stft"
    normalize: bool = False
    signal_augment: bool = False
    spec_augment: bool = False


class AudioFeatureExtractor:
    def __init__(
        self, audio_conf: AudioFeaturesConfig,
    ):
        super().__init__()
        self.feature_type = audio_conf.feature_type
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = NAME2WINDOWTYPE[audio_conf.window]
        self.normalize = audio_conf.normalize
        self.signal_augment = audio_conf.signal_augment
        self.spec_augment = audio_conf.spec_augment

        if self.feature_type == "mfcc":
            self.mfcc = torchaudio.transforms.MFCC(
                sample_rate=SAMPLE_RATE, n_mfcc=get_feature_dim(audio_conf)
            )
        elif self.feature_type == "mel":
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_mels=get_feature_dim(audio_conf)
            )
        if hasattr(self, "audio_text_files"):
            self.audio_files = [
                f for f, _ in self.audio_text_files
            ]  # TODO accessing the child-classes attribute!!

    def process(self, audio_path):
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
        spect = calc_stft_librosa(
            y, self.sample_rate, self.window_size, self.window_stride, self.window
        )
        if self.spec_augment:
            spect = spec_augment(spect)

        return spect


class DataConfig(NamedTuple):
    dirs: List[str]
    labels: List[str]
    min_len: float = 1  # seconds
    max_len: float = 20  # seconds


from corpora.librispeech import generate_audiofile_text_tuples

MILLISECONDS_TO_SECONDS = 0.001


def get_length(audio_file):
    si, ei = torchaudio.info(audio_file)
    return si.length / si.channels / si.rate


class Sample(NamedTuple):
    audio_file: str
    text: str
    length: float  # in seconds


class SpectrogramDataset(Dataset):
    def __init__(
        self, conf: DataConfig, audio_conf: AudioFeaturesConfig,
    ):
        self.conf = conf
        samples_g = (
            Sample(audio_file, text, get_length(audio_file))
            for f in conf.dirs
            for audio_file, text in generate_audiofile_text_tuples(f)
        )
        samples_g = filter(
            lambda s: s.length > conf.min_len and s.length > conf.max_len, samples_g
        )
        sorted_samples = sorted(samples_g, key=lambda s: s.length)
        self.samples = sorted_samples
        self.size = len(self.samples)
        self.labels_map = dict([(labels[i], i) for i in range(len(conf.labels))])
        self.audio_fe = AudioFeatureExtractor(audio_conf)
        super().__init__()

    def __getitem__(self, index):
        s: Sample = self.samples[index]
        feat = self.audio_fe.process(s.audio_file)
        transcript = self.parse_transcript(s.text)
        return feat, transcript

    def parse_transcript(self, transcript: str) -> List[int]:
        transcript = list(
            filter(None, [self.labels_map.get(x) for x in list(transcript)])
        )  # TODO(tilo) like this it erases unknown letters
        # transcript = [self.labels_map.get(x,UNK) for x in list(transcript)] # TODO(tilo) better like this?
        return transcript

    def __len__(self):
        return self.size


if __name__ == "__main__":
    # fmt: off
    labels = ["_", "'","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," "]
    # fmt: on

    HOME = os.environ["HOME"]
    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"

    conf = DataConfig([raw_data_path + "/dev-other"], labels)
    audio_conf = AudioFeaturesConfig()
    train_dataset = SpectrogramDataset(conf, audio_conf)
    datum = train_dataset[0]
    print()
