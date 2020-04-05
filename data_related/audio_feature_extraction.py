import math
from tempfile import NamedTemporaryFile
from typing import List, NamedTuple

import scipy
import torch
import torchaudio

from data_related.audio_util import load_audio
from data_related.data_augmentation.signal_augment import augment_with_sox
from data_related.data_augmentation.spec_augment import spec_augment
from data_related.feature_extraction import calc_stft_librosa


SAMPLE_RATE = 16_000


class AudioFeaturesConfig(NamedTuple):
    sample_rate: int = 16_000
    feature_type: str = "stft"
    normalize: bool = False
    signal_augment: bool = False
    spec_augment: bool = False


def get_feature_dim(conf: AudioFeaturesConfig):
    feature_type = conf.feature_type
    if feature_type == "mfcc":
        FEATURE_DIM = 40
    elif feature_type == "mel":
        FEATURE_DIM = 161
    elif feature_type == "stft":
        # FEATURE_DIM = int(
        #     math.floor((conf.sample_rate * conf.window_size) / 2) + 1
        # )  # 161 #TODO(tilo)
        FEATURE_DIM = 161
    else:
        assert False
    return FEATURE_DIM


def augment_and_load(original_audio_file: str, audio_files: List[str]):
    """
    :param original_audio_file:
    :param audio_files: used for signal-inference-noise
    :return:
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        while True:
            try:
                augment_with_sox(original_audio_file, audio_files, augmented_filename)
                audio = load_audio(augmented_filename)
                break
            except:
                pass

    return audio


class AudioFeatureExtractor:
    def __init__(self, audio_conf: AudioFeaturesConfig, audio_files):
        super().__init__()
        self.audio_files = audio_files
        self.feature_type = audio_conf.feature_type
        self.sample_rate = audio_conf.sample_rate
        self.normalize = audio_conf.normalize
        self.signal_augment = audio_conf.signal_augment
        self.spec_augment = audio_conf.spec_augment
        self.feature_dim = get_feature_dim(audio_conf)

        if self.feature_type == "mfcc":
            self.mfcc = torchaudio.transforms.MFCC(
                sample_rate=SAMPLE_RATE, n_mfcc=self.feature_dim
            )
        elif self.feature_type == "mel":
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_mels=self.feature_dim
            )

    def process(self, audio_path):
        if self.signal_augment:
            y = augment_and_load(audio_path, self.audio_files)
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

        NAME2WINDOWTYPE = {
            "hamming": scipy.signal.hamming,
            "hann": scipy.signal.hann,
            "blackman": scipy.signal.blackman,
            "bartlett": scipy.signal.bartlett,
        }

        window_size: float = 0.02
        window_stride: float = 0.01
        window = NAME2WINDOWTYPE["hamming"]

        spect = calc_stft_librosa(
            y, self.sample_rate, window_size, window_stride, window
        )
        if self.spec_augment:
            spect = spec_augment(spect)

        return spect


def get_length(audio_file):
    si, ei = torchaudio.info(audio_file)
    return si.length / si.channels / si.rate
