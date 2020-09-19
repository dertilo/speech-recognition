from sagemaker.s3 import S3Uploader
from util.util_methods import exec_command

if __name__ == '__main__':
    s3_path ="s3://tilos-ml-bucket/LibriSpeech"
    local_path = "/home/tilo/data/asr_data/ENGLISH/LibriSpeech"
    folder_name = "dev-clean-some_preprocessed"
    file_to_upload = f"/tmp/{folder_name}.tar.gz"
    exec_command(f"cd {local_path} && tar -czvf {file_to_upload} {folder_name}")
    S3Uploader.upload(file_to_upload, f"{s3_path}")