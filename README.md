# Speech-Recognition

## based on [NeMo](https://github.com/NVIDIA/NeMo)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dertilo/speech-recognition/blob/master/nemo_asr/nemo.ipynb)

## based on [espnet](https://github.com/espnet/espnet)
* [no batch inference yet?](https://github.com/espnet/espnet/issues/2186)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dertilo/speech-recognition/blob/master/espnet_lightning/espnet.ipynb)

## based on [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
PyTorch implementation of [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) trained with the CTC objective.
### differences to [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch.git)
* no use of [warp-ctc](https://github.com/SeanNaren/warp-ctc.git), instead [torch.nn.CTCLoss](https://pytorch.org/docs/master/generated/torch.nn.CTCLoss.html)
* powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

#### [results](https://app.wandb.ai/dertilo/speech-recognition/runs/28gqsg3l/overview?workspace=user-)
* after 8 epochs and 24hours with Adam
```shell script
python evaluation.py --model epoch=8.ckpt --datasets test-clean
2528 of 2620 samples are suitable for training
100%|█████████████████████████████████████| 127/127 [02:12<00:00,  1.04s/it]
Test Summary    Average WER 9.925       Average CER 3.239

python evaluation.py --model epoch=8.ckpt --datasets test-other
2893 of 2939 samples are suitable for training
100%|███████████████████████████████████████| 145/145 [01:19<00:00,  1.83it/s]
Test Summary    Average WER 27.879      Average CER 11.739
```


## Datasets
### Librispeech
1. to download data see: https://github.com/dertilo/speech-to-text/corpora/download_corpora.py
* splits
    ```
    datasets = [
        ("train", ["train-clean-100", "train-clean-360", "train-other-500"]),
        ("eval", ["dev-clean", "dev-other"]),
        ("test", ["test-clean", "test-other"]),
    ]
    ```
* number of samples
    ```
    train got 281241 samples
    eval got 5567 samples
    test got 5559 samples
    ```
