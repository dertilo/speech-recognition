from pprint import pprint
from espnet2.bin.main import run_espnet
import argparse

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    output_data_dir = os.environ["SM_OUTPUT_DATA_DIR"]
    checkpoint_path = output_data_dir + "/checkpoints/"
    input_path = os.environ["SM_CHANNEL_TRAINING"]
    # fmt:off
    parser.add_argument('--gpus', type=int, default=1) # used to support multi-GPU or CPU training
    parser.add_argument('--config', type=str,default="minimal_config.yml") # used to support multi-GPU or CPU training
    # fmt:on
    args = parser.parse_args()
    pprint(args.__dict__)

    dev_name = "dev-clean-some_preprocessed"
    cmd = "tar xzf %s -C %s" % (input_path + "/" + dev_name + ".tar.gz", input_path)
    assert os.system(cmd) == 0
    print(os.listdir(input_path))

    os.environ["LRU_CACHE_CAPACITY"] = str(1)

    dev_path = f"{input_path}/{dev_name}"
    run_espnet(
        train_path=dev_path,
        valid_path=dev_path,
        out_path=output_data_dir,
        config=args.config,
        num_gpus=args.gpus,
    )
