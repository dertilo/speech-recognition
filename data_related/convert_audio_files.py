import multiprocessing
from concurrent import futures
from concurrent.futures.thread import ThreadPoolExecutor

import torchaudio
from tqdm import tqdm
from util.util_methods import process_with_threadpool, exec_command

from data_related.datasets.librispeech import read_librispeech

torchaudio.set_audio_backend("sox")

def convert_flac_to_mp3(flac_file: str, base_folder, dataset):
    file_name = (
        flac_file.replace(dataset + "/", "").replace("/", "_").replace(".flac", ".mp3")
    )
    mp3_file = f"{base_folder}/dev-clean-mp3/{file_name}"
    # x,fs = torchaudio.backend.sox_backend.load(flac_file)
    # torchaudio.backend.sox_backend.save(mp3_file,x,sample_rate=fs)
    exec_command(f"sox {flac_file} {mp3_file}")

if __name__ == "__main__":
    base_folder = "/home/tilo/data/asr_data/ENGLISH/LibriSpeech"
    dataset = "%s/dev-clean" % base_folder
    file2utt = read_librispeech(dataset)

    input = [
        {"flac_file": f, "base_folder": base_folder, "dataset": dataset}
        for f in file2utt.keys()
    ]
    # convert_flac_to_mp3(**input[0])
    r = process_with_threadpool(
        input,
        convert_flac_to_mp3,
        max_workers=2,
    )
    list(tqdm(r))

    # 2703it [00:36, 74.15it/s] with 10 workers
    # 2703it [01:08, 39.27it/s] with 2 workers
