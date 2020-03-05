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


def build_sox_distortions(
    audio_file, gain=0, tempo=1.0, pitch=0, reverb=0, lowpass=8000, highpass=1
):
    params = {
        "gain -n ": gain,
        "tempo": tempo,
        "pitch": pitch,
        "reverb": (reverb, 50, 100, 100, 0, 0),
        "lowpass": lowpass,
        "highpass": highpass,
    }
    param_str = " ".join([k + " " + to_str(v) for k, v in params.items()])
    sox_params = "sox {} -p {} ".format(audio_file, param_str)
    return sox_params


def build_sox_noise(
    audio_file,
    amod_lowpass_cutoff=0.1,
    lowpass_cutoff=1.0,
    highpass_cutoff=1.0,
    noise_gain=-4,
):

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


def add_signals_trim_to_len(original, signals):
    signals_to_add = " ".join(["<(%s)" % s for s in signals])
    sox_cmd = "sox -m {signals} -p trim 0 $(soxi -D {original})".format(
        signals=signals_to_add, original=original
    )
    return sox_cmd


def random_augmentation(original_file, audio_files, augmented_file):
    # interfere_files = librosa.util.find_files(interfere_path)
    interfere_file = np.random.choice(audio_files)

    lowpass = int(round(np.random.uniform(low=1000, high=8000)))
    highpass = int(round(np.random.uniform(low=1, high=lowpass - 1000)))

    signal_gain = round(np.random.uniform(low=-30, high=-1), 2)
    signal_params = {
        "gain": signal_gain,
        "tempo": round(np.random.uniform(low=0.7, high=1.4), 2),
        "pitch": int(round(np.random.uniform(low=-200, high=100))),
        "reverb": int(round(np.random.uniform(low=0, high=50))),
        "lowpass": lowpass,
        "highpass": highpass,
    }

    lowpass = int(round(np.random.uniform(low=50, high=8000)))
    highpass = int(round(np.random.uniform(low=1, high=lowpass)))

    interfere_params = {
        "gain": round(np.random.uniform(low=-50, high=signal_gain - 15), 2),
        "tempo": round(np.random.uniform(low=0.6, high=1.4), 2),
        "pitch": int(round(np.random.uniform(low=-500, high=500))),
        "reverb": int(round(np.random.uniform(low=0, high=100))),
        "lowpass": lowpass,
        "highpass": highpass,
    }

    # params = {'signal_params':signal_params,'interfere_params':interfere_params,'noise_power':noise_power}
    # pprint(params)

    signal = build_sox_distortions(original_file, **signal_params)
    interfere_signal = build_sox_distortions(interfere_file, **interfere_params)

    noise_power = round(np.random.uniform(-60, -30), 2)
    lowpass = int(round(np.random.uniform(low=100, high=8000)))
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

    sox_pipe = add_signals_trim_to_len(original_file, [signal, noise, interf])
    sox_cmd = sox_pipe + " > " + augmented_file
    subprocess.call(["bash", "-c", sox_cmd])


if __name__ == "__main__":
    original = "/tmp/original.wav"
    augmented = "/tmp/augmented.wav"
    interfering = "/tmp/interfere2.wav"

    # signal = build_sox_distortions(original, gain=-10, tempo=0.7, pitch=100, reverb=0)
    # interfere_signal = build_sox_distortions(interfering, gain=-20, tempo=0.8, pitch=100, reverb=50)
    # noise = build_sox_noise(original, noise_gain=-5,lowpass_cutoff=100,highpass_cutoff=10)
    # interf = build_sox_interference(interfering, interfere_signal)
    #
    # sox_pipe = add_signals_trim_to_len(original,[signal,noise,interf])
    # sox_cmd = sox_pipe +' > '+augmented
    # subprocess.call(["bash", "-c", sox_cmd])

    interfere_path = "/home/tilo/data/asr_data/SPANISH/openslr_spanish/es_co_female"
    random_augmentation(original, interfere_path, augmented)
