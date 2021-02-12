from time import time

from corpora.common import maybe_extract, AudioConfig, process_write_manifest
from corpora.common_voice import build_audiofile2text

if __name__ == "__main__":

    malpaudios = [
        "common_voice_es_19499901.mp3",
        "common_voice_es_19499893.mp3",
    ]  # broken audios

    audio_config = AudioConfig("mp3")

    ac = f"{audio_config.format}{'' if audio_config.bitrate is None else '_' + str(audio_config.bitrate)}"
    corpus_name = "SPANISH_CV"

    raw_zipfile = "/data/es.tar.gz"
    work_dir = "/data"
    corpus_dir = f"{work_dir}/{corpus_name}"
    raw_dir = f"{corpus_dir}/raw"
    maybe_extract(raw_zipfile, raw_dir)

    for split_name in ["train", "dev", "test"]:
        processed_corpus_dir = f"{corpus_dir}/{split_name}_processed_{ac}"
        file2utt = build_audiofile2text(raw_dir, split_name,"es",broken_files=malpaudios)
        print(f"beginn processing {processed_corpus_dir}")
        start = time()
        process_write_manifest((raw_dir, processed_corpus_dir), file2utt, audio_config)
        print(f"processing done in: {time() - start} secs")

"""
beginn processing /data/SPANISH_CV/train_processed_mp3
161811it [43:20, 62.22it/s]   
beginn processing /data/SPANISH_CV/dev_processed_mp3
15089it [04:16, 58.78it/s]   
beginn processing /data/SPANISH_CV/test_processed_mp3
15089it [04:16, 58.93it/s]   
"""