### singularity image

* locally build singularity image

        sudo singularity build deepspeech.sif container.def

* copy to hpc

    scp /home/tilo/code/SPEECH/deepspeech.pytorch/deepspeech.simg tilo-himmelsbach@gateway.hpc.tu-berlin.de:/home/users/t/tilo-himmelsbach/

### on hpc
    module load singularity/2.5.2
    singularity shell --nv deepspeech.sif
    
#### evaluation
    python test.py --model-path librispeech_pretrained_v2.pth --test-manifest data/libri_test_clean.csv --cuda --half

#### transcribing 
    python transcribe.py --model-path librispeech_models/libri_full_final.pth --audio-path LibriSpeech_dataset/train/wav/4133-6541-0035.wav
    
# RUNS

librispeech-clean-100

    python -m multiproc train.py --train-manifest libri_train100_manifest.csv --val-manifest libri_val_manifest.csv --id libri_100_new

librispeech-clean-100 new on gpu006

python -m multiproc train.py --train-manifest libri_train100_manifest.csv --log-dir tensorboard_logdir/libri_100_new --hidden-layers 5 --opt-level O1 --loss-scale 1 --id libri_100_new --checkpoint --save-folder librispeech_save/100_new
python -m multiproc train.py --continue-from librispeech_save/100_new/deepspeech_1.pth.tar --train-manifest libri_train100_manifest.csv --log-dir tensorboard_logdir/libri_100_new --hidden-layers 5 --opt-level O1 --loss-scale 1 --id libri_100_new --checkpoint --save-folder librispeech_save/100_new

for debug
    
    python -m multiproc train.py --log-dir tensorboard_logdir/debug --train-manifest libri_train_manifest_some.csv --val-manifest libri_train_manifest_some.csv --hidden-layers 2 --opt-level O1 --loss-scale 1 --id debug --checkpoint --save-folder librispeech_save/debug --model-path librispeech_models/deepspeech_debug.pth
    python train.py --log-dir tensorboard_logdir/debug --train-manifest libri_train_manifest_some.csv --val-manifest libri_train_manifest_some.csv --hidden-layers 2 --opt-level O1 --loss-scale 1 --id debug --checkpoint --save-folder librispeech_save/debug --model-path librispeech_models/deepspeech_debug.pth

### spanish
* debug
    python -m multiproc train.py --labels-path spanish_vocab.json --train-manifest spanish_some.csv --val-manifest spanish_some.csv --id debug

* locally on laptop
    python train.py --train-manifest spanish_train_laptop.csv --val-manifest spanish_eval_laptop.csv --id debug --hidden-layers 2 --hidden-size 128 --num-workers 2

* full
    python -m multiproc train.py --id spanish
    
* mel

 hung up forever with this 

        Epoch: [3][301/1469]    Time 1.925 (1.839)      Data 0.002 (0.064)      Loss 48.2371 (48.2371)
        WARNING: received a nan loss, setting loss value to 0
        Skipping grad update

    python -m multiproc train.py --batch-size 60 --feature-type mel --continue-from checkpoints/spanish_mel/deepspeech_2.pth.tar --labels-path spanish_vocab.json --id spanish_mel
    python -m multiproc train.py --feature-type mel --id spanish_mel

* augmented

    python -m multiproc train.py --feature-type stft --continue-from checkpoints/spanish_augmented/deepspeech_5.pth.tar --augment --labels-path spanish_vocab.json --train-manifest spanish_train_manifest.csv --val-manifest spanish_eval_manifest.csv --id spanish_augmented

#### transcribing
    
    python transcribe_manifest.py --model-path librispeech_save/spanish/deepspeech_2.pth.tar --manifest spanish_eval_manifest.csv

# TODO
* data-augmentation gain+tempo doing any good?
    * ja it seems to do so
* adding noise to signal doing any good? noise generation


ffplay -f lavfi 'amovie=original.wav, asplit [a][out1];[a] showspectrum=mode=separate:color=intensity:slide=1:scale=cbrt [out0]'