import subprocess

import librosa
import numpy as np

def to_str(v):
    if isinstance(v, tuple):
        s = " ".join(str(x) for x in v)
    elif isinstance(v, float) or isinstance(v, int):
        s = str(v)
    else:
        assert False

    return s


def build_sox_distortions(audio_file, gain=0, tempo=1.0, pitch=0, reverb=0,lowpass=8000,highpass=1):
    params = {
        "gain -n ": gain,
        "tempo": tempo,
        "pitch": pitch,
        "reverb": (reverb, 50, 100, 100, 0, 0),
        "lowpass":lowpass,
        "highpass":highpass,
    }
    param_str = " ".join([k + " " + to_str(v) for k, v in params.items()])
    sox_params = "sox {} -p {} ".format(audio_file, param_str)
    return sox_params


def build_sox_noise(audio_file, lowpass_cutoff=1, noise_gain=-4):
    params = {"lowpass_cutoff": lowpass_cutoff, "noise_gain": noise_gain}

    sox_params = "sox {audio_file} -p synth whitenoise lowpass {lowpass_cutoff} synth whitenoise amod gain -n {noise_gain}".format(
        audio_file=audio_file, **params
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
    interfere_file,interfere_signal, lowpass_cutoff=1, ac_gain=-6
):
    factor = build_varying_amplitude_factor(interfere_file, lowpass_cutoff, ac_gain)
    return multiply_signals(factor, interfere_signal)


def add_signals_trim_to_len(original, signals):
    signals_to_add = " ".join(["<(%s)" % s for s in signals])
    sox_cmd = "sox -m {signals} -p trim 0 $(soxi -D {original})".format(
        signals=signals_to_add, original=original
    )
    return sox_cmd

def random_augmentation(original_file, interfere_path, augmented_file):
    interfere_files = librosa.util.find_files(interfere_path)
    interfere_file = np.random.choice(interfere_files)

    lowpass = int(round(np.random.uniform(low=500, high=8000)))
    highpass = int(round(np.random.uniform(low=1, high=lowpass-400)))

    signal_params = {
        'gain':round(np.random.uniform(low=-6, high=-1),2),
        'tempo':round(np.random.uniform(low=0.6, high=1.4),2),
        'pitch':int(round(np.random.uniform(low=-500, high=500))),
        'reverb':int(round(np.random.uniform(low=0, high=100))),
        'lowpass':lowpass,
        'highpass':highpass,
    }

    lowpass = int(round(np.random.uniform(low=100, high=8000)))
    highpass = int(round(np.random.uniform(low=1, high=lowpass)))

    interfere_params = {
        'gain':round(np.random.uniform(low=-24, high=-3),2),
        'tempo':round(np.random.uniform(low=0.6, high=1.4),2),
        'pitch':int(round(np.random.uniform(low=-500, high=500))),
        'reverb':int(round(np.random.uniform(low=0, high=100))),
        'lowpass': lowpass,
        'highpass': highpass,
    }
    noise_power = np.random.uniform(-30,-1)

    signal = build_sox_distortions(original_file, **signal_params)
    interfere_signal = build_sox_distortions(interfere_file, **interfere_params)
    noise = build_sox_noise(original,np.random.uniform(0.1,2) ,noise_power)

    interf = build_sox_interference(interfere_file, interfere_signal,
                                    lowpass_cutoff=np.random.uniform(0.5,2),
                                    ac_gain=int(round(np.random.uniform(-9,-3))))

    sox_pipe = add_signals_trim_to_len(original,[signal,noise,interf])
    sox_cmd = sox_pipe +' > '+augmented_file
    subprocess.call(["bash", "-c", sox_cmd])


if __name__ == "__main__":
    original = "/tmp/original.wav"
    augmented = "/tmp/augmented.wav"
    # interfering = "/tmp/interfere2.wav"
    #
    # signal = build_sox_distortions(original, gain=-6, tempo=0.6, pitch=-500, reverb=0)
    # interfere_signal = build_sox_distortions(interfering, gain=-4, tempo=0.8, pitch=100, reverb=50)
    # noise = build_sox_noise(original, noise_gain=-6)
    # interf = build_sox_interference(interfering, interfere_signal)
    #
    # sox_pipe = add_signals_trim_to_len(original,[signal,noise,interf])
    # sox_cmd = sox_pipe +' > '+augmented
    # subprocess.call(["bash", "-c", sox_cmd])


    interfere_path = '/home/tilo/data/asr_data/SPANISH/openslr_spanish/es_co_female'
    random_augmentation(original,interfere_path,augmented)
