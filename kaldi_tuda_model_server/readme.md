## kaldi-model-server

#### setup
```shell
git clone git@github.com:dertilo/kaldi-model-server.git
git checkout tilo
cd kaldi-model-server/docker

docker-compose -f docker-compose.yml up -d --build #rebuild an up
# OR
docker-compose build -f docker-compose.yml # just build
docker-compose up -d

docker exec -it kamose bash
```
#### model
* must be in `kaldi-model-server/model`

#### data
* scp-file build by [build_scp_file.py](build_scp_file.py)

#### run inference
* inside docker-container
```shell
python nnet3_model.py -i scp:/docker-share/data/asr_data/GERMAN/german-speechdata-package-v2/wav_test_Yamaha.scp
```