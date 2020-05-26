import os
from warnings import filterwarnings

from data_related.librispeech import build_dataset
from lightning.lightning_model import LitSTTModel
from lightning.litutil import generic_train, build_args

filterwarnings("ignore")

if __name__ == "__main__":
    p = {
        "save_path": "/tmp/mlflow_logs",
        "n_gpu":0,
        "hidden_layers":2,
        "hidden_size":64,
        "num_workers":0
    }
    args = build_args(LitSTTModel,p)

    data_path = os.environ["HOME"] + "/data/asr_data/"

    train_dataset = build_dataset()
    args.vocab_size = len(train_dataset.char2idx)
    # BLANK_INDEX = train_dataset.char2idx[BLANK_SYMBOL]
    args.audio_feature_dim = train_dataset.audio_fe.feature_dim

    model = LitSTTModel(args)

    generic_train(model, args)
