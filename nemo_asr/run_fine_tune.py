import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import DictConfig
import pathlib
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
# stolen from https://gist.github.com/gtcooke94/89d933cda31ee75fec3c32e295b5b718
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(
    model_name="QuartzNet15x5Base-En"
)

train_manifest_file = "</path/to/train-manifest>"
val_manifest_file = "</path/to/val-manifest>"

trainer = pl.Trainer(gpus=1, max_epochs=20)


new_opt = {
    "betas": [0.95, 0.25],
    "lr": 0.001,
    "name": "novograd",
    "sched": {
        "last_epoch": -1,
        "min_lr": 0.0,
        "monitor": "val_loss",
        "name": "CosineAnnealing",
        "reduce_on_plateau": False,
        "warmup_ratio": 0.12,
        "warmup_steps": None,
    },
    "weight_decay": 0.001,
}



train_ds_config = {
    "batch_size": 16,
    "is_tarred": False,
    "labels": [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ],
    "manifest_filepath": str(train_manifest_file),
    "max_duration": 16.7,
    "sample_rate": 16000,
    "shuffle": True,
    "tarred_audio_filepaths": None,
    "trim_silence": True,
}

train_ds_config = DictConfig(train_ds_config)

val_ds_config = {
    "batch_size": 16,
    "labels": [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ],
    "manifest_filepath": str(val_manifest_file),
    "sample_rate": 16000,
    "shuffle": False,
}

val_ds_config = DictConfig(val_ds_config)


quartznet.setup_training_data(train_data_config=train_ds_config)
quartznet.setup_validation_data(val_data_config=val_ds_config)
quartznet.set_trainer(trainer)
quartznet.setup_optimization(optim_config=DictConfig(new_opt))

trainer.fit(quartznet)
quartznet.save_to("</path/to/save/to>")
