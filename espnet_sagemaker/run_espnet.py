from pprint import pprint

import argparse

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt:off
    output_data_dir = os.environ['SM_OUTPUT_DATA_DIR']
    checkpoint_path = output_data_dir + "/checkpoints/"
    parser.add_argument('-o','--output-data-dir', type=str, default=output_data_dir)
    parser.add_argument('--data_dir', type=str,default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path)
    # fmt:on
    args = parser.parse_args()
    pprint(args.__dict__)

    dev_name = "dev-clean-some_preprocessed"
    assert os.system("tar xzf %s -C %s" % (args.data_dir + "/" + dev_name + ".tar.gz", args.data_dir)) == 0
    print(os.listdir(args.data_dir))

    os.environ["LRU_CACHE_CAPACITY"] = str(1)

    from espnet2.bin.main import run_espnet
    dev_path = f"{args.data_dir}/{dev_name}"
    run_espnet(train_path=dev_path,
               valid_path=dev_path,
               config="minimal_config.yml")
