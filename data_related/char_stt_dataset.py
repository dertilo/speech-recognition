import os

from torch.distributed import get_rank
from torch.utils.data import Dataset
from typing import NamedTuple, List, Dict

from tqdm import tqdm
from util import data_io

from data_related.audio_feature_extraction import (
    AudioFeaturesConfig,
    AudioFeatureExtractor,
    get_length,
)
from utils import HOME


class DataConfig(NamedTuple):
    labels: List[str]
    min_len: float = 1  # seconds
    max_len: float = 20  # seconds


MILLISECONDS_TO_SECONDS = 0.001


class Sample(NamedTuple):
    audio_file: str
    text: str
    length: float  # in seconds


def sort_samples_in_corpus(corpus, min_len, max_len) -> List[Sample]:
    print("sort samples by length")
    print("rank: %d" % get_rank(), flush=True)
    samples_g = (
        Sample(audio_file, text, get_length(audio_file))
        for audio_file, text in tqdm(corpus.items())
    )
    samples_g = filter(lambda s: s.length > min_len and s.length < max_len, samples_g)
    samples: List[Sample] = sorted(samples_g, key=lambda s: s.length)
    assert len(samples) > 0
    print("%d of %d samples are suitable for training" % (len(samples), len(corpus)))
    return samples


def load_samples(file: str):
    def process(d):
        d["audio_file"] = d["audio_file"].replace("/docker-share", HOME)
        return Sample(**d)

    return [process(d) for d in data_io.read_jsonl(file)]


class CharSTTDataset(Dataset):
    def __init__(
        self,
        conf: DataConfig,
        audio_conf: AudioFeaturesConfig,
        corpus: Dict[str, str] = None,
        corpus_file=None,
    ):
        assert corpus or corpus_file
        self.conf = conf
        if corpus_file is not None:
            self.samples = load_samples(corpus_file)
        else:
            self.samples = sort_samples_in_corpus(corpus, conf.min_len, conf.max_len)
        self.size = len(self.samples)

        self.char2idx = dict([(conf.labels[i], i) for i in range(len(conf.labels))])
        self.audio_fe = AudioFeatureExtractor(
            audio_conf, [s.audio_file for s in self.samples]
        )
        super().__init__()

    def __getitem__(self, index):
        s: Sample = self.samples[index]
        feat = self.audio_fe.process(s.audio_file)
        transcript = self.parse_transcript(s.text)
        return feat, transcript

    def parse_transcript(self, transcript: str) -> List[int]:
        transcript = list(
            filter(None, [self.char2idx.get(x) for x in list(transcript)])
        )  # TODO(tilo) like this it erases unknown letters
        # transcript = [self.labels_map.get(x,UNK) for x in list(transcript)] # TODO(tilo) better like this?
        return transcript

    def __len__(self):
        return self.size


if __name__ == "__main__":
    # fmt: off
    labels = ["_", "'","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," "]
    # fmt: on
    from corpora.librispeech import librispeech_corpus

    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"

    conf = DataConfig(labels)
    audio_conf = AudioFeaturesConfig()
    corpus = {
        k: v
        for p in [raw_data_path + "/dev-other"]
        for k, v in librispeech_corpus(p).items()
    }
    train_dataset = CharSTTDataset(corpus=corpus, conf=conf, audio_conf=audio_conf)
    datum = train_dataset[0]
    print()
