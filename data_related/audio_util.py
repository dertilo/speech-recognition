from typing import NamedTuple

import numpy
import torchaudio


class Sample(NamedTuple):
    audio_file: str
    text: str
    length: float  # in seconds

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
