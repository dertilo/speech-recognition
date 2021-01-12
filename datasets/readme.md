# Datasets
* [collection-repo by JRMeyer](https://github.com/JRMeyer/open-speech-corpora)

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

### Tuda-corpus

* broken files: (see `some-where/kaldi-tuda-de/s5_r2/local/tuda_files_to_skip.txt`)
```shell
2014-03-27-11-50-33
2014-06-17-13-46-27
2014-03-24-13-39-24
2014-08-27-11-05-29
2015-02-10-13-45-07
2015-01-27-11-31-41
2014-08-05-11-08-34
2014-03-18-15-28-52
2014-03-18-15-29-23
```