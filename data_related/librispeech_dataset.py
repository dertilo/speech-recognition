import math
import os
import subprocess
from tempfile import NamedTemporaryFile

import scipy.signal
import torch
import torchaudio
from torch.utils.data import Dataset

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

windows = {
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
        self, audio_conf, normalize=False, signal_augment=False, spec_augment=False,
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
        if hasattr(self, "audio_text_files"):
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
        spect = calc_stft_librosa(
            y, self.sample_rate, self.window_size, self.window_stride, self.window
        )
        if self.spec_augment:
            spect = spec_augment(spect)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset):
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

    HOME = os.environ["HOME"]
    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"

    manifest_file = raw_data_path +'/dev-clean_manifest.jsonl'
    train_dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=manifest_file,
        labels=labels,
        normalize=True,
        signal_augment=False,
        spec_augment=False,
    )

    datum = train_dataset[0]
    print()
