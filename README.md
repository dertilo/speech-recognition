# Speech-Recognition
PyTorch implementation of [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) trained with the CTC objective.
### heavily inspired by https://github.com/SeanNaren/deepspeech.pytorch.git
### differences to [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch.git)
* no use of [warp-ctc](https://github.com/SeanNaren/warp-ctc.git), instead [torch.nn.CTCLoss](https://pytorch.org/docs/master/generated/torch.nn.CTCLoss.html)
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
* if on __hpc-node:__ do: `module load nvidia/cuda/10.1 && module load comp`
* install it: `git clone https://github.com/NVIDIA/apex && cd apex && OMP_NUM_THREADS=8 pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

## train
* on frontend:
```shell script
OMP_NUM_THREADS=2 wandb init
```
* on hpc
    ```shell script
    module load comp
    export PYTHONPATH=$HOME/SPEECH/speech-to-text:$HOME/SPEECH/speech-recognition:$HOME/UTIL/util:$HOME/SPEECH/fairseq
    WANDB_MODE=dryrun python train.py
    ```
## evaluate
```shell script
python evaluation.py --model libri_960_1024_32_11_04_2020/deepspeech_9.pth.tar --datasets test-clean
```
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