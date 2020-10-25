# [NeMo](https://github.com/NVIDIA/NeMo)
### Info
* [pretrained models](https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels)
* rclone , synchronize to colab
```
rclone sync -P --exclude ".git/**" --exclude ".idea/**" --exclude "build/**" --exclude "*.pyc" --max-size 100k $HOME/code/SPEECH/NeMo dertilo-googledrive:NeMo
```

### Questions
* why is `preprocessor` and `spec_augmentation` done within `forward`? why not in dataloader?
* why not black formatted?
* no sortish sampler or bucketing? "simply" take huggingface's `DistributedSortishSampler`
* why soundfile which is unable to read mp3 ? 
* what about `environment.yml` ?

### TODO
* evaluate the pretrained `QuartzNet15x5Base-En`: wav+soundfile vs. mp3+torchaudio
* unicode for manifests: `json.dump(metadata, f, ensure_ascii=False)`