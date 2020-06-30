# Speech-Recognition
PyTorch implementation of [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) trained with the CTC objective.
### heavily inspired by https://github.com/SeanNaren/deepspeech.pytorch.git
### differences to [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch.git)
* no use [warp-ctc](https://github.com/SeanNaren/warp-ctc.git), instead [pytorch implementation](https://pytorch.org/docs/master/generated/torch.nn.CTCLoss.html)
* powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

### Datasets
#### Librispeech
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
## setup
### install [apex](https://github.com/NVIDIA/apex)
* if on __hpc-node: __ do: `module load nvidia/cuda/10.1 && module load comp`
* install it: `git clone https://github.com/NVIDIA/apex && cd apex && OMP_NUM_THREADS=8 pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

## train
* on hpc
    ```shell script
    module load comp
    export PYTHONPATH=$HOME/SPEECH/speech-to-text:$HOME/SPEECH/speech-recognition:$HOME/UTIL/util:$HOME/SPEECH/fairseq
    python train.py
    ```
## evalute
```shell script
python evaluation.py --model libri_960_1024_32_11_04_2020/deepspeech_9.pth.tar --datasets test-clean
```
#### results
* after 10 epochs and 24hours with Adam
```shell script
    test-clean    Average WER 9.042       Average CER 3.038
    test-other    Average WER 26.233      Average CER 11.346
```