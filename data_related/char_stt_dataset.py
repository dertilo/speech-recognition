from torch.distributed import get_rank
from torch.utils.data import Dataset
from typing import NamedTuple, List

from data_related.audio_feature_extraction import (
    AudioFeaturesConfig,
    AudioFeatureExtractor,
    AUDIOFEATUREEXTRACTORS, )
from data_related.utils import Sample
from utils import HOME


class DataConfig(NamedTuple):
    labels: List[str]
    min_len: float = 1  # seconds
    max_len: float = 20  # seconds


MILLISECONDS_TO_SECONDS = 0.001


def sort_samples_in_corpus(samples: List[Sample], min_len, max_len) -> List[Sample]:
    f_samples_g = filter(lambda s: s.duration > min_len and s.duration < max_len, samples)
    s_samples: List[Sample] = sorted(f_samples_g, key=lambda s: s.duration)
    assert len(s_samples) > 0
    print("%d of %d samples are suitable for training" % (len(s_samples), len(samples)))
    return s_samples


class CharSTTDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        conf: DataConfig,
        audio_conf: AudioFeaturesConfig,
    ):
        self.conf = conf
        self.samples = sort_samples_in_corpus(samples, conf.min_len, conf.max_len)

        self.char2idx = dict([(conf.labels[i], i) for i in range(len(conf.labels))])
        self.audio_fe: AudioFeatureExtractor = AUDIOFEATUREEXTRACTORS[
            audio_conf.feature_type
        ](audio_conf, [s.audio_file for s in self.samples])
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
        return len(self.samples)


if __name__ == "__main__":
    from data_related.datasets.librispeech import build_librispeech_corpus

    # fmt: off
    labels = ["_", "'","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," "]
    # fmt: on

    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"

    conf = DataConfig(labels)
    audio_conf = AudioFeaturesConfig()

    samples = build_librispeech_corpus(
        HOME + "/data/asr_data/ENGLISH/LibriSpeech", "eval", ["dev-clean", "dev-other"]
    )

    train_dataset = CharSTTDataset(samples, conf=conf, audio_conf=audio_conf)
    datum = train_dataset[0]
    print()
