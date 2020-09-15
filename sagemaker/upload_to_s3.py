from sagemaker.s3 import S3Uploader
from util.util_methods import exec_command

if __name__ == '__main__':
    s3_path ="s3://tilos-ml-bucket/LibriSpeech"
    file_name = 'dev-clean-mp3.tar.gz'
    local_path = "/home/tilo/data/asr_data/ENGLISH/LibriSpeech"
    folder_name = "preprocessed_dev-clean"
    file_to_upload = f"/tmp/{folder_name}.tar.gz"
    exec_command(f"tar -czvf {file_to_upload} {local_path}/{folder_name}")
    S3Uploader.upload(file_to_upload, f"{s3_path}/{folder_name}.tar.gz")