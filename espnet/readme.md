# [espnet](https://github.com/espnet/espnet)
* setup: 
    * build image: `docker build -t espnet -f Dockerfile_espnet .`
    * run container: `docker run -it --rm -v local_folder:/docker-share espnet:latest bash`

* [finetuning](https://espnet.github.io/espnet/tutorial.html#how-to-use-finetuning)

### data processing
* what are segments ?
* what is this `wav.scp` good for? look just like a list of files
### run.sh
* pretrained espnet models
* train in docker
* `--stage 2 --stop-stage 6`