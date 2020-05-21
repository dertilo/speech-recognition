from datetime import datetime

from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.sparse import (
    csr_matrix,
)  # TODO(tilo): if not imported before torch on HPC-cluster it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

from src.model.transformer.lightning_model_eng import LightningModel
import torch.backends.cudnn as cudnn
import random
import torch as t
import os
import argparse
from warnings import filterwarnings
from pytorch_lightning.logging.test_tube_logger import TestTubeLogger
filterwarnings('ignore')


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-epochs', default=200, type=int)
    parser = LightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()

def setup_testube_logger(save_dir) -> TestTubeLogger:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    return TestTubeLogger(
        save_dir=save_dir,
        version=0,
        name="litstt-"+dt_string,
    )

if __name__ == '__main__':
    hparams = get_args()
    data_path = os.environ['HOME']+'/data/asr_data/'
    model = LightningModel(hparams)

    random.seed(hparams.seed)
    t.manual_seed(hparams.seed)
    cudnn.deterministic = True


    logger = setup_testube_logger(data_path)
    checkpoint = ModelCheckpoint(filepath=data_path+'/checkpoints/litstt/',
                                 monitor='val_mer', save_top_k=-1)
    trainer = Trainer(
        logger=logger,
        checkpoint_callback=checkpoint,
        # fast_dev_run=True,
        # overfit_pct=0.03,
        # profiler=True,
        val_check_interval=1.0,
        log_save_interval=100,
        row_log_interval=10,
        gpus=1,
        precision=16,
        distributed_backend='dp',
        max_nb_epochs=hparams.epochs,
        gradient_clip_val=5.0,
        use_amp=True,
        amp_level='O1',
        nb_sanity_val_steps=0,
        log_gpu_memory='all'
    )
    # if hparams.evaluate:
    #     trainer.run_evaluation()
    # else:
    trainer.fit(model)
