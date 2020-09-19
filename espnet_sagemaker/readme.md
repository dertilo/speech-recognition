* [build container](https://github.com/aws/deep-learning-containers/blob/master/custom_images.md)

* `--force-reinstall` (see Dockerfile) makes Dockerfile practically non-appendable (cause it reinstalls big+fat torch!!), thus produces a base-image from which one can inherit
```shell script
docker build -f Dockerfile_base -t 706022464121.dkr.ecr.eu-central-1.amazonaws.com/pytorch-espnet:1.6.0-cpu-py3-base .
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 706022464121.dkr.ecr.eu-central-1.amazonaws.com/pytorch-espnet
docker push 706022464121.dkr.ecr.eu-central-1.amazonaws.com/pytorch-espnet:1.6.0-cpu-py3-base
```