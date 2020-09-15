# MNIST on SageMaker with PyTorch Lightning
import json
import boto3
import sagemaker
import wandb
from sagemaker.pytorch import PyTorch
# based on: https://github.com/aletheia/mnist_pl_sagemaker/blob/master/main.py
source_dir = '../'
wandb.sagemaker_auth(path=source_dir)

sagemaker_session = sagemaker.Session()
s3_path = "s3://tilos-ml-bucket/LibriSpeech"

# The IAM Role which SageMaker will impersonate to run the estimator
# Remember you cannot use sagemaker.get_execution_role()
# if you're not in a SageMaker notebook, an EC2 or a Lambda
# (i.e. running from your local PC)

role = 'arn:aws:iam::706022464121:role/SageMakerRole_MNIST' # sagemaker.get_execution_role()

# Creates a new PyTorch Estimator with params
estimator = PyTorch(
  # name of the runnable script containing __main__ function (entrypoint)
  entry_point='train.py',
  # path of the folder containing training code. It could also contain a
  # requirements.txt file with all the dependencies that needs
  # to be installed before running
  source_dir=source_dir,
  role=role,
  framework_version='1.4.0',
  py_version="py3",
  instance_count=1,
  # instance_type="local",# 'ml.p2.xlarge',
  instance_type="ml.c5.xlarge",#"ml.g4dn.xlarge",# 'ml.p2.xlarge',
  # these hyperparameters are passed to the main script as arguments and
  # can be overridden when fine tuning the algorithm
  use_spot_instances = True,
  max_wait = 24 * 60 * 60, # seconds; see max_run
  # checkpoint_s3_uri = ...
  hyperparameters={
  'max_epochs': 2,
  'batch_size': 32,
  })

# Call fit method on estimator, wich trains our model, passing training
# and testing datasets as environment variables. Data is copied from S3
# before initializing the container

#
 #sagemaker-eu-central-1-706022464121.s3.eu-central-1.amazonaws.com/pytorch-training-2020-09-04-15-01-18-109/output/output.tar.gz
estimator.fit(s3_path)

# [ml.p2.xlarge, ml.m5.4xlarge, ml.m4.16xlarge, ml.c5n.xlarge, ml.p3.16xlarge, ml.m5.large, ml.p2.16xlarge, ml.c4.2xlarge, ml.c5.2xlarge, ml.c4.4xlarge, ml.c5.4xlarge, ml.c5n.18xlarge, ml.g4dn.xlarge, ml.g4dn.12xlarge, ml.c4.8xlarge, ml.g4dn.2xlarge, ml.c5.9xlarge, ml.g4dn.4xlarge, ml.c5.xlarge, ml.g4dn.16xlarge, ml.c4.xlarge, ml.g4dn.8xlarge, ml.c5n.2xlarge, ml.c5n.4xlarge, ml.c5.18xlarge, ml.p3dn.24xlarge, ml.p3.2xlarge, ml.m5.xlarge, ml.m4.10xlarge, ml.c5n.9xlarge, ml.m5.12xlarge, ml.m4.xlarge, ml.m5.24xlarge, ml.m4.2xlarge, ml.p2.8xlarge, ml.m5.2xlarge, ml.p3.8xlarge, ml.m4.4xlarge]