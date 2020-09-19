import sagemaker
import wandb
from sagemaker.pytorch import PyTorch

source_dir = '.'
wandb.sagemaker_auth(path=source_dir)

sagemaker_session = sagemaker.Session()
s3_path = "s3://tilos-ml-bucket/LibriSpeech"

role = "arn:aws:iam::706022464121:role/service-role/AmazonSageMaker-ExecutionRole-20200317T145654"
# Creates a new PyTorch Estimator with params
estimator = PyTorch(
  entry_point='run_espnet.py',
  source_dir=source_dir,
  role=role,
  # framework_version='1.6.0',
  # py_version="py3",
  image_uri="706022464121.dkr.ecr.eu-central-1.amazonaws.com/pytorch-espnet:1.6.0-cpu-py3",
  instance_count=1,
  instance_type="local",# 'ml.p2.xlarge',
  # instance_type="ml.c5.xlarge",#"ml.g4dn.xlarge",# 'ml.p2.xlarge',
  # use_spot_instances = True,
  # max_wait = 24 * 60 * 60, # seconds; see max_run
  # checkpoint_s3_uri = ...
  hyperparameters={
  })

estimator.fit(s3_path)
