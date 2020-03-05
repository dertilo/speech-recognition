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


if __name__ == "__main__":
    """
    play original.wav tempo 1.4 gain -9 pitch -100 reverb 50 80 100 10 0 0
    """

    original = "/tmp/original.wav"
    augmented = "/tmp/augmented.wav"

    sox_pipe = build_sox_distortions(original,gain=-4,tempo=1.2,pitch=-200,reverb=50)
    subprocess.call(["bash", "-c", sox_pipe+' > '+augmented])
