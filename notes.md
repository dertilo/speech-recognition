
## deepspeech.pytorch
### install [apex](https://github.com/NVIDIA/apex)
* if on __hpc-node:__ do: `module load nvidia/cuda/10.1 && module load comp`
* install it: `git clone https://github.com/NVIDIA/apex && cd apex && OMP_NUM_THREADS=8 pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

### train
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
### evaluate
```shell script
python evaluation.py --model libri_960_1024_32_11_04_2020/deepspeech_9.pth.tar --datasets test-clean
```

### on hpc
`module load comp`
`export PYTHONPATH=$HOME/SPEECH/speech-to-text:$HOME/SPEECH/speech-recognition:$HOME/UTIL/util`
`python lightning/train_lightning.py`
