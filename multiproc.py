import sys
import subprocess

import torch

WORLD_SIZE = torch.cuda.device_count()
workers = []

for i in range(WORLD_SIZE):
    argslist = ['train.py']

    argslist.append("--world-size")
    argslist.append(str(WORLD_SIZE))

    argslist.append("--rank")
    argslist.append(str(i))

    argslist.append("--gpu-rank")
    argslist.append(str(i))

    stdout = None if i == 0 else open("GPU_" + str(i) + ".log", "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout, stderr=stdout)
    workers.append(p)

for p in workers:
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(returncode=p.returncode, cmd=p.args)
