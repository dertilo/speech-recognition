import os
import subprocess

import librosa
import numpy as np
from tqdm import tqdm
from typing import Dict

MAX_FREQ = 7999


def to_str(v):
    if isinstance(v, tuple):
        s = " ".join(str(x) for x in v)
    elif isinstance(v, float) or isinstance(v, int):
        s = str(v)
    else:
        assert False

    return s


def build_sox_distortions(audio_file, params):
    param_str = " ".join([k + " " + to_str(v) for k, v in params.items()])
    sox_params = "sox {} -p {} ".format(audio_file, param_str)
    return sox_params


def build_sox_noise(
    audio_file,
    amod_lowpass_cutoff=0.1,
    lowpass_cutoff=MAX_FREQ,
    highpass_cutoff=1,
    noise_gain=-4,
):
    '''
    play original.wav synth whitenoise lowpass 0.1 synth whitenoise amod gain -n 0 lowpass 100 highpass 1
    '''

    sox_params = "sox {audio_file} -p synth whitenoise lowpass {amod_lowpass_cutoff} synth whitenoise amod gain -n {noise_gain} lowpass {lowpass_cutoff} highpass {highpass_cutoff}".format(
        audio_file=audio_file,
        amod_lowpass_cutoff=amod_lowpass_cutoff,
        lowpass_cutoff=lowpass_cutoff,
        highpass_cutoff=highpass_cutoff,
        noise_gain=noise_gain,
    )
    return sox_params


def build_varying_amplitude_factor(audio_file, lowpass_cutoff=1, ac_gain=-9):
    ac = "sox {} -p synth whitenoise lowpass {} gain -n {}".format(
        audio_file, lowpass_cutoff, ac_gain
    )
    dc = "sox {} -p gain -90 dcshift 0.5".format(audio_file)
    return "sox -m <({}) <({}) -p".format(ac, dc)


def multiply_signals(signal_a, signal_b):
    return ("sox -T <({signal_a}) <({signal_b}) -p").format(
        signal_a=signal_a, signal_b=signal_b,
    )


def build_sox_interference(
    interfere_file, interfere_signal, lowpass_cutoff=1, ac_gain=-6
):
    factor = build_varying_amplitude_factor(interfere_file, lowpass_cutoff, ac_gain)
    return multiply_signals(factor, interfere_signal)


def add_signals_trim_to_len(original, signals, augmented):
    signals_to_add = " ".join(["<(%s)" % s for s in signals])
    sox_cmd = "sox -m {signals} {augmented} trim 0 $(soxi -D {original})".format(
        signals=signals_to_add, original=original, augmented=augmented
    )
    return sox_cmd


def build_random_bandpass(min_low=50, min_band_width=100) -> Dict:
    d = {}
    max_high_cutoff = MAX_FREQ
    if np.random.choice([True, False]):
        lowpass = int(round(np.random.uniform(low=min_low, high=MAX_FREQ)))
        d["lowpass"] = lowpass
        max_high_cutoff = lowpass - min_band_width

    if np.random.choice([True, False]):
        highpass = int(round(np.random.uniform(low=1, high=max_high_cutoff)))
        d["highpass"] = highpass

    return d


def random_augmentation(original_file, audio_files, augmented_file):
    interfere_file = np.random.choice(audio_files)
    min_SNR = 20
    min_SIR = 10

    signal_gain = round(np.random.uniform(low=-30, high=-1), 2)
    signal_params = {
        "gain": signal_gain,
        "tempo": round(np.random.triangular(left=0.8,mode=1.0,right=1.2), 2),
        "pitch": int(round(np.random.triangular(left=-100,mode=0,right=100))),
        "reverb": (int(round(np.random.uniform(low=0, high=50))), 50, 100, 100, 0, 0),
    }
    signal_params.update(build_random_bandpass(1000, 1000))

    interfere_params = {
        "gain": round(np.random.uniform(low=-50, high=signal_gain - min_SIR), 2),
        "tempo": round(np.random.uniform(low=0.6, high=1.4), 2),
        "pitch": int(round(np.random.uniform(low=-500, high=500))),
        "reverb": (int(round(np.random.uniform(low=0, high=100))), 50, 100, 100, 0, 0),
    }
    signal_params.update(build_random_bandpass(50, 100))

    # params = {'signal_params':signal_params,'interfere_params':interfere_params,'noise_power':noise_power}
    # pprint(params)

    signal = build_sox_distortions(original_file, signal_params)
    interfere_signal = build_sox_distortions(interfere_file, interfere_params)

    noise_power = round(np.random.uniform(-60, signal_gain - min_SNR), 2)
    lowpass = int(round(np.random.uniform(low=100, high=MAX_FREQ)))
    highpass = int(round(np.random.uniform(low=1, high=lowpass)))
    noise = build_sox_noise(
        original_file, np.random.uniform(0.1, 2), lowpass, highpass, noise_power
    )

    interf = build_sox_interference(
        interfere_file,
        interfere_signal,
        lowpass_cutoff=np.random.uniform(0.5, 2),
        ac_gain=int(round(np.random.uniform(-9, -3))),
    )

    sox_cmd = add_signals_trim_to_len(
        original_file, [signal, noise, interf], augmented_file
    )
    FNULL = open(os.devnull, "w")
    subprocess.call(["bash", "-c", sox_cmd], stdout=FNULL, stderr=subprocess.STDOUT)
    # subprocess.call(["bash", "-c", sox_cmd])
    # output = subprocess.check_output(["bash", "-c", sox_cmd])
    # if len(output)>0 and 'FAIL' in output:
    #     print(output)
    # return 1 if len(output)>0 else 0


def augment_with_specific_params():
    signal_gain = 0
    signal = build_sox_distortions(
        original, dict(gain=signal_gain, tempo=1.0, pitch=100, reverb=0, lowpass=9000)
    )
    interfere_signal = build_sox_distortions(
        interfering, dict(gain=signal_gain-10, tempo=0.8, pitch=100, reverb=50)
    )
    noise = build_sox_noise(
        original, noise_gain=signal_gain-60, lowpass_cutoff=6000, highpass_cutoff=10
    )
    interf = build_sox_interference(interfering, interfere_signal)
    sox_cmd = add_signals_trim_to_len(original, [signal, noise, interf], augmented)
    subprocess.call(["bash", "-c", sox_cmd])


if __name__ == "__main__":
    original = "/tmp/original.wav"
    augmented = "/tmp/augmented.wav"
    interfering = "/tmp/interfere2.wav"

    # augment_with_specific_params()

    #
    for k in range(9):
        random_augmentation(original, [interfering], "/tmp/augmented_%d.wav"%k)
    # assert False
    # path = os.environ['HOME']+"/data/asr_data/SPANISH"
    # audio_files = librosa.util.find_files(path)

    #
    # with open('spanish_train_manifest.csv') as f:
    #     audio_text_files = f.readlines()
    # audio_files = [x.strip().split(",")[0] for x in audio_text_files]
    #
    # for k in tqdm(range(100000)):
    #     original = np.random.choice(audio_files)
    #     random_augmentation(original, audio_files, augmented)
