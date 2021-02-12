import argparse
import os
import sys

import yaml

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

if __name__ == "__main__":
    """
    dev-other WER for QuartzNet5x5 should be close to 15.69%; see: NeMo/nemo/collections/asr/models/ctc_models.py
    """
    base_path = ""
    config_path = f"{base_path}/code/NeMo/examples/asr/conf/config.yaml"
    with open(config_path, "r") as stream:
        cfg = yaml.safe_load(stream)

    dummy_manifest = "/tmp/dummy_train.json"
    manifest = f"{base_path}/data/dev_other_wav.json"
    assert 0 == os.system(f"head -n 10 {manifest} > {dummy_manifest}")
    cfg["model"]["train_ds"]["manifest_filepath"] = dummy_manifest
    cfg["model"]["validation_ds"]["manifest_filepath"] = manifest
    cfg["trainer"]["max_epochs"] = 1
    cfg["trainer"]["gpus"] = 1

    from pytorch_lightning.loggers import WandbLogger

    logger = WandbLogger(name="test", project="nemo")
    print(cfg["trainer"].pop("logger"))
    trainer = pl.Trainer(logger=logger, **cfg["trainer"])

    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
        model_name="QuartzNet15x5Base-En"
    )
    asr_model.setup_training_data(train_data_config=cfg["model"]["train_ds"])
    asr_model.setup_validation_data(val_data_config=cfg["model"]["validation_ds"])

    print(f"num trainable params: {asr_model.num_weights}")
    trainer.fit(asr_model)

    """
    wandb: Run summary:
    wandb:           train_loss 80.0228
    wandb:        learning_rate 0.01
    wandb:   training_batch_wer 0.64789
    wandb:                epoch 0
    wandb:             val_loss 24.55768
    wandb:              val_wer 0.1616
    wandb:                _step 0
    wandb:             _runtime 42
    """
