### gunther
    rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=.git --exclude=data --max-size=1m /home/tilo/code/SPEECH/speech-recognition gunther@gunther:/home/gunther/tilo_data/SPEECH/
    docker build -t deepspeech .
    docker run --shm-size 8G --runtime=nvidia --rm -it -v /home/gunther/tilo_data:/docker-share --net=host --env JOBLIB_TEMP_FOLDER=/tmp/ deepspeech:latest bash
    export PYTHONPATH=$HOME/SPEECH/speech-to-text:$HOME/SPEECH/speech-recognition:$HOME/UTIL/util
preprocess data on gunther
    `python data_related/librispeech.py`
copy to hpc, because preprocessing on hpc is way to slow because of slow beegfs
    `cp ~/gunther/data/asr_data/ENGLISH/LibriSpeech/*.jsonl.gz ~/hpc/data/asr_data/ENGLISH/LibriSpeech/`
### on hpc
`module load comp`
`export PYTHONPATH=$HOME/SPEECH/speech-to-text:$HOME/SPEECH/speech-recognition:$HOME/UTIL/util`
`python lightning/train_lightning.py`
