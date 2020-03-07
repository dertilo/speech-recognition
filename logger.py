import os

import torch


def to_np(x):
    return x.cpu().numpy()


class TensorBoardLogger(object):
    def __init__(self, id, log_dir, log_params):
        os.makedirs(log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter

        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir, max_queue=1, flush_secs=30)
        self.log_params = log_params

    def update(self, epoch, values, parameters=None):
        to_log = {
            "train-loss": values["loss_results"][epoch],
            "valid-loss": values["loss_eval_results"][epoch],
            "WER": values["wer_results"][epoch],
            "CER": values["cer_results"][epoch],
        }
        print(to_log)

        for k,v in to_log.items():
            self.tensorboard_writer.add_scalars(k, {self.id:v}, epoch)

        if self.log_params:
            for tag, value in parameters():
                tag = tag.replace(".", "/")
                self.tensorboard_writer.add_histogram(tag, to_np(value), epoch)
                self.tensorboard_writer.add_histogram(
                    tag + "/grad", to_np(value.grad), epoch
                )