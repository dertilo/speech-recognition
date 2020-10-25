# attempt to power espnet-asr with pytorch-lightning
* possible to make `NumElementsBatchSampler` compatible with pytorch-lightning which recommends to [remove samplers](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html) ?
* huggingface-transformers use `DistributedSortishSampler`

* sync to google-drive
```shell script
rclone sync -P --exclude ".git/**" --exclude ".idea/**" --exclude "build/**" --exclude "*.pyc" --max-size 100k $HOME/code/SPEECH/speech-recognition dertilo-googledrive:speech-recognition
```
