import argparse
import os
import sys
# supposed to be run on google-colab
nemo_path = "/mydrive/NeMo"
if nemo_path not in sys.path:
  sys.path.append(nemo_path)
print(sys.path)

import yaml

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl


parser = argparse.ArgumentParser(description="evaluate pretrained")
parser.add_argument("--manifest", required=True, default="/content/dev_other.json", type=str)
parser.add_argument("--name", required=True, default="debug", type=str)
parser.add_argument("--gpus", required=False, default=0, type=int)
args = parser.parse_args()

if __name__ == '__main__':

    config_path = '/mydrive/NeMo/examples/asr/conf/config.yaml'
    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    dummy_manifest = "/content/dummy_train.json"
    assert 0==os.system(f"head -n 10 {args.manifest} > {dummy_manifest}")
    cfg["model"]["train_ds"]["manifest_filepath"]= dummy_manifest
    cfg["model"]["validation_ds"]["manifest_filepath"]= args.manifest
    cfg["trainer"]["max_epochs"]=1
    cfg["trainer"]["gpus"]=args.gpus

    from pytorch_lightning.loggers import WandbLogger

    logger = WandbLogger(name=args.name, project="nemo")
    print(cfg["trainer"].pop("logger"))
    trainer = pl.Trainer(logger=logger,**cfg["trainer"])

    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet5x5LS-En")
    asr_model.setup_training_data(train_data_config=cfg["model"]['train_ds'])
    asr_model.setup_validation_data(val_data_config=cfg["model"]['validation_ds'])

    print(f"num trainable params: {asr_model.num_weights}")
    sys.stdout.flush()
    trainer.fit(asr_model)