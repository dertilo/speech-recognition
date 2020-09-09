import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

if __name__ == '__main__':

    d = ModelDownloader()
    model_name = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
    loaded = d.download_and_unpack(model_name)
    speech2text = Speech2Text(**loaded)
    speech2text.tokenizer.model=f"{loaded['asr_model_file'].split('/exp')[0]}/{speech2text.tokenizer.model}"
    path = "/home/dertilo/data/asr_data/ENGLISH/LibriSpeech/dev-other/LibriSpeech/dev-other/116/288045"
    file_name = "116-288045-0000.flac"
    speech, rate = soundfile.read(f"{path}/{file_name}")
    nbests = speech2text(speech)
    text, *_ = nbests[0]
    print(text)