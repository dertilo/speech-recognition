# [espnet](https://github.com/espnet/espnet)
* setup: 
    * build image: `docker build -t espnet -f Dockerfile_espnet .`
    * run container: `docker run -it --rm -v local_folder:/docker-share espnet:latest bash`


### data processing

### run.sh
* `--stage 2 --stop-stage 6`
* `spk2utt` -> text file of speaker to utterance-list map