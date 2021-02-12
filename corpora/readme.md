# Here everything concerning dataset specific preprocessing
* downloading, text-cleaning, audio-format-conversion; independent of training models!

# Datasets
* [collection-repo by JRMeyer](https://github.com/JRMeyer/open-speech-corpora)
* [MLS: Multilingual LibriSpeech](https://indico2.conference4me.psnc.pl/event/35/contributions/3585/attachments/1060/1101/Wed-2-6-10.pdf)

### CommonVoice Spanish
* processing ~236k samples took ~1 hour
```shell
236312it [1:07:57, 57.96it/s]   
```
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

