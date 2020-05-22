# based on https://github.com/SeanNaren/deepspeech.pytorch.git

Implementation of DeepSpeech2 for PyTorch. Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture, trained with the CTC activation function.

### Datasets
#### Librispeech
1. to download data see: `https://github.com/dertilo/speech-to-text/corpora/download_corpora.py`
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


## Training

#### results
* after 10 epochs and 24hours with Adam
```shell script
    test-clean    Average WER 8.936       Average CER 2.962
    test-other    Average WER 25.961      Average CER 11.192
```


### Docker

```bash
DOCKER_SHARE=<some path>
docker build -t deepspeech .
docker run --shm-size 8G --runtime=nvidia --rm -it -v $DOCKER_SHARE:/docker-share --net=host --env JOBLIB_TEMP_FOLDER=/tmp/ deepspeech:latest bash
```
