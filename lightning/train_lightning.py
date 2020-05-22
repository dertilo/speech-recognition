import argparse
import os
from warnings import filterwarnings

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer

from lightning.lightning_model import LitSTTModel, Params, build_dataset
from utils import BLANK_SYMBOL

filterwarnings("ignore")

if __name__ == "__main__":
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-epochs", default=200, type=int)
    parser = LitSTTModel.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

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
            batch_size=64,
            num_workers=4,
        )
    )
    # random.seed(hparams.seed)
    # t.manual_seed(hparams.seed)
    # cudnn.deterministic = True

    checkpoint = ModelCheckpoint(
        filepath=data_path + "/checkpoints/litstt/", monitor="val_mer", save_top_k=-1
    )

    trainer = Trainer(
        logger=True,
        checkpoint_callback=checkpoint,
        val_check_interval=1.0,
        gpus=2,
        precision=16,
        distributed_backend="ddp",
        max_nb_epochs=10,
        gradient_clip_val=5.0,
        use_amp=True,
        amp_level="O1",
    )
    trainer.fit(model)
