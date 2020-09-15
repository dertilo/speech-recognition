from data_related.audio_feature_extraction import AudioFeaturesConfig
from data_related.datasets.librispeech import build_dataset, LibrispeechDataModule
from lightning.litutil import build_args, generic_train
import os

from lightning.lit_deepspeech import LitDeepSpeech

if __name__ == '__main__':
    data_path = os.environ["HOME"] + "/data/asr_data"
    # p = {
    #     "exp_name": "vggtransformer",
    #     # "exp_name": "debug",
    #     "run_name": "some run",
    #     "save_path": data_path + "/mlruns",
    #     "batch_size": 8,
    #     # "fp16": True,# any value sets it to True
    #     "n_gpu": 2,
    #     "enc_output_dim": 512,
    #     "vggblock_enc_config": "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]",
    #     "transformer_enc_config": "((1024, 8, 4096, True, 0.15, 0.15, 0.15),) * 5",
    #     "num_workers": 4,
    #     "max_epochs": 10,
    # }
    #
    # args = build_args(LitVGGTransformerEncoder, p)
    #
    # train_dataset = build_dataset()
    # args.vocab_size = len(train_dataset.char2idx)
    # # BLANK_INDEX = train_dataset.char2idx[BLANK_SYMBOL]
    # args.input_feat_per_channel = train_dataset.audio_fe.feature_dim
    #
    # model = LitVGGTransformerEncoder(args)
    #
    # generic_train(model, args)


    p = {
        "exp_name": "deepspeech-librispeech-100",
        "run_name": "train-100-stft-161-normalized",
        "save_path": data_path,
        "batch_size": 32,
        "fp16": "True",
        "n_gpu": 2,
        "num_workers": 4,
        "max_epochs": 10,
    }
    args = build_args(LitDeepSpeech, p)

    audio_conf = AudioFeaturesConfig(feature_type="stft")
    args.audio_feature_dim = audio_conf.feature_dim

    model = LitDeepSpeech(args)

    ldm = LibrispeechDataModule(
        os.environ["HOME"] + "/data/asr_data/ENGLISH/LibriSpeech",
        collate_fn=model._collate_fn,
        hparams=argparse.Namespace(**{"num_workers": 0, "batch_size": 8}),
    )

    generic_train(model, args)
