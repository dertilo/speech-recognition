import soundfile
import torchaudio
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import os

if __name__ == '__main__':

    path = "/home/tilo/data/asr_data/ENGLISH/dev-other/700/122867"
    file_name = "700-122867-0026.flac"
    audio_file = f"{path}/{file_name}"

    si, _ = torchaudio.info(audio_file)
    normalize_denominator = 1 << si.precision
    speech, rate = torchaudio.backend.sox_backend.load(
        audio_file, normalization=normalize_denominator
    )
    speech = speech.t()

    d = ModelDownloader(cachedir=os.environ["HOME"]+"/data/")
    model_name = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
    loaded = d.download_and_unpack(model_name)
    speech2text = Speech2Text(**loaded)
    speech2text.tokenizer.model=f"{loaded['asr_model_file'].split('/exp')[0]}/{speech2text.tokenizer.model}"

    nbests = speech2text(speech)
    text, *_ = nbests[0]
    print(text)