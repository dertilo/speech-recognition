import os
from espnet2.bin.main import run_espnet

if __name__ == "__main__":
    os.environ["LRU_CACHE_CAPACITY" ] = str(1)
    run_espnet()
