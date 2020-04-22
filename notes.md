### gunther

    rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=.git --exclude=data --max-size=1m /home/tilo/code/SPEECH/speech-recognition gunther@gunther:/home/gunther/tilo_data/SPEECH/
    docker build -t deepspeech .
    docker run --shm-size 8G --runtime=nvidia --rm -it -v /home/gunther/tilo_data:/docker-share --net=host --env JOBLIB_TEMP_FOLDER=/tmp/ deepspeech:latest bash
    export PYTHONPATH=$HOME/SPEECH/speech-to-text:$HOME/SPEECH/speech-recognition:$HOME/UTIL/util

    python -m multiproc train.py

### on hpc
#### singularity image

* locally build singularity image

        sudo singularity build deepspeech.sif container.def

* copy to hpc

    scp /home/tilo/code/SPEECH/deepspeech.pytorch/deepspeech.simg tilo-himmelsbach@gateway.hpc.tu-berlin.de:/home/users/t/tilo-himmelsbach/

    rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=.git --exclude=data --max-size=1m /home/tilo/code/SPEECH tilo-himmelsbach@gateway.hpc.tu-berlin.de:/home/users/t/tilo-himmelsbach/

#### setup  
  
    module load singularity/2.5.2
    singularity shell --nv deepspeech.sif
    export PYTHONPATH=$HOME/SPEECH/speech-to-text:$HOME/SPEECH/speech-recognition:$HOME/UTIL/util

#### train 
    python -m multiproc train.py --id libri_960_1024_32_11_04_2020
    
#### evaluate

    python evaluation.py --model libri_960_1024_32_11_04_2020/deepspeech_9.pth.tar --datasets test-clean 

evaluate original 
    
    python test.py --model-path ~/data/asr_data/checkpoints/librispeech_pretrained_v2.pth --test-manifest libri_test_other_manifest.csv --cuda --half

# RUNS

* batch-size 50 and lr = 2e-3 does not converge + crashes


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