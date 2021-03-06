import librosa
import numpy as np
from abc import abstractmethod

import numpy
from tempfile import NamedTemporaryFile
from typing import List, NamedTuple

import scipy
import torch
import torchaudio

from data_related.data_augmentation.signal_augment import augment_with_sox
from data_related.data_augmentation.spec_augment import spec_augment


def load_audio(audio_file: str, target_rate=16_000) -> numpy.ndarray:
    si, _ = torchaudio.info(audio_file)
    normalize_denominator = 1 << si.precision
    sound, sample_rate = torchaudio.load(
        audio_file, normalization=normalize_denominator
    )
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_rate
        )
        sound = resampler(sound)

    y = sound.squeeze().numpy()
    return y

class AudioFeaturesConfig(NamedTuple):
    sample_rate: int = 16_000
    feature_type: str = "stft"
    normalize: bool = True
    signal_augment: bool = False
    spec_augment: bool = False

    @property
    def feature_dim(self):
        feature_type = self.feature_type
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


def augment_and_load(original_audio_file: str, audio_files: List[str]) -> numpy.ndarray:
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
    def __init__(self, audio_conf: AudioFeaturesConfig, audio_files: List[str]):
        self.audio_files = audio_files
        self.audio_conf = audio_conf

    def process(self, audio_file: str) -> torch.Tensor:
        if self.audio_conf.signal_augment:
            y = augment_and_load(audio_file, self.audio_files)
        else:
            y = load_audio(audio_file)
        return self._extract_features(y)

    @abstractmethod
    def _extract_features(self, sig: numpy.ndarray) -> torch.Tensor:
        raise NotImplementedError


class TorchAudioExtractor(AudioFeatureExtractor):
    def __init__(self, audio_conf: AudioFeaturesConfig, audio_files: List[str]):
        if audio_conf.feature_type == "mfcc":
            self.extractor = torchaudio.transforms.MFCC(
                sample_rate=self.audio_conf.sample_rate,
                n_mfcc=self.audio_conf.feature_dim,
            )
        elif audio_conf.feature_type == "mel":
            self.extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.audio_conf.sample_rate,
                n_mels=self.audio_conf.feature_dim,
            )

        super().__init__(audio_conf, audio_files)

    def _extract_features(self, sig: numpy.ndarray) -> torch.Tensor:
        torch_tensor = torch.from_numpy(sig).unsqueeze(0)
        return self.extractor.forward(torch_tensor).data.squeeze(0)


def calc_stft_librosa(y,sample_rate,window_size,window_stride,window):
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    # STFT
    D = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    return spect

class LibrosaExtractor(AudioFeatureExtractor):
    def _extract_features(self, sig: numpy.ndarray) -> torch.Tensor:
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
            sig, self.audio_conf.sample_rate, window_size, window_stride, window
        )
        if self.audio_conf.spec_augment:
            spect = spec_augment(spect)

        return spect


AUDIOFEATUREEXTRACTORS = {
    "mfcc": TorchAudioExtractor,
    "mel": TorchAudioExtractor,
    "stft": LibrosaExtractor,
}


def get_length(audio_file):
    si, ei = torchaudio.info(audio_file)
    return si.duration / si.channels / si.rate
