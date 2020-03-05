import subprocess


def to_str(v):
    if isinstance(v, tuple):
        s = " ".join(str(x) for x in v)
    elif isinstance(v, float) or isinstance(v, int):
        s = str(v)
    else:
        assert False

    return s


def build_sox_distortions(audio_file, gain=0, tempo=1.0, pitch=0, reverb=0):
    params = {
        "gain": gain,
        "tempo": tempo,
        "pitch": pitch,
        "reverb": (reverb, 50, 100, 100, 0, 0),
    }
    param_str = " ".join([k + " " + to_str(v) for k, v in params.items()])
    sox_params = "sox {} -p {} ".format(audio_file, param_str)
    return sox_params


def build_sox_noise(audio_file, lowpass_cutoff=1, noise_gain=-4):
    params = {"lowpass_cutoff": lowpass_cutoff, "noise_gain": noise_gain}

    sox_params = "sox {audio_file} -p synth whitenoise lowpass {lowpass_cutoff} synth whitenoise amod gain {noise_gain}".format(
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
    interfere_file, interfere_gain=-12, lowpass_cutoff=1, ac_gain=-6
):
    factor = build_varying_amplitude_factor(interfere_file, lowpass_cutoff, ac_gain)
    signal = "sox {} -p gain {} reverse".format(interfere_file, interfere_gain)
    return multiply_signals(factor, signal)


def add_signals_trim_to_len(original, signals):
    signals_to_add = " ".join(["<(%s)" % s for s in signals])
    sox_cmd = "sox -m {signals} -p trim 0 $(soxi -D {original})".format(
        signals=signals_to_add, original=original
    )
    return sox_cmd

if __name__ == "__main__":
    """
    play original.wav tempo 1.4 gain -9 pitch -100 reverb 50 80 100 10 0 0
    """

    original = "/tmp/original.wav"
    augmented = "/tmp/augmented.wav"
    interfere = "/tmp/interfere.wav"

    signal = build_sox_distortions(original, gain=-4, tempo=1.2, pitch=-200, reverb=50)
    noise = build_sox_noise(original, noise_gain=0)
    interf = build_sox_interference(interfere)

    sox_pipe = add_signals_trim_to_len(original,[signal,noise,interf])
    sox_cmd = sox_pipe +' > '+augmented
    print(sox_cmd)
    subprocess.call(["bash", "-c", sox_cmd])
