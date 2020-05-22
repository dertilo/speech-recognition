from datetime import datetime

from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.sparse import (
    csr_matrix,
)  # TODO(tilo): if not imported before torch on HPC-cluster it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

import torch.backends.cudnn as cudnn
import random
import torch as t
import os
import argparse
from warnings import filterwarnings
from pytorch_lightning.logging.test_tube_logger import TestTubeLogger

from data_related.audio_feature_extraction import AudioFeaturesConfig
from data_related.char_stt_dataset import DataConfig, CharSTTDataset
from data_related.librispeech import LIBRI_VOCAB, build_librispeech_corpus
from lightning.lightning_model import LitSTTModel, Params, build_dataset
from utils import BLANK_SYMBOL

filterwarnings("ignore")


def setup_testube_logger(save_dir) -> TestTubeLogger:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    return TestTubeLogger(save_dir=save_dir, version=0, name="litstt-" + dt_string,)


if __name__ == "__main__":
    # parent_parser = argparse.ArgumentParser(add_help=False)
    # parent_parser.add_argument('-epochs', default=200, type=int)
    # parser = LitSTTModel.add_model_specific_args(parent_parser)
    # hparams = parser.parse_args()

    data_path = os.environ["HOME"] + "/data/asr_data/"

    train_dataset = build_dataset()
    vocab_size = len(train_dataset.char2idx)
    BLANK_INDEX = train_dataset.char2idx[BLANK_SYMBOL]
    audio_feature_dim = train_dataset.audio_fe.feature_dim

    model = LitSTTModel(
        Params(
            hidden_size=1024,
            hidden_layers=5,
            audio_feature_dim=audio_feature_dim,
            vocab_size=vocab_size,
            batch_size=32,
            num_workers=4,
        )
    )
    #
    # trainer = Trainer(logger=False, checkpoint_callback=False, max_epochs=3)
    # trainer.fit(litmodel)

    # random.seed(hparams.seed)
    # t.manual_seed(hparams.seed)
    # cudnn.deterministic = True

    # logger = setup_testube_logger(data_path)
    # checkpoint = ModelCheckpoint(filepath=data_path+'/checkpoints/litstt/',
    #                              monitor='val_mer', save_top_k=-1)
    trainer = Trainer(
        logger=False,
        checkpoint_callback=False,
        # fast_dev_run=True,
        # overfit_pct=0.03,
        # profiler=True,
        val_check_interval=1.0,
        gpus=1,
        precision=16,
        distributed_backend="dp",
        max_nb_epochs=1,
        gradient_clip_val=5.0,
        use_amp=True,
        amp_level="O1",
        nb_sanity_val_steps=0,
        # log_gpu_memory='all'
    )
    trainer.fit(model)
