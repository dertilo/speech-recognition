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
        loss, wer, cer = (
            values["loss_results"][epoch],
            values["wer_results"][epoch],
            values["cer_results"][epoch],
        )
        values = {
            "Avg Train Loss": loss,
            "Avg Valid Loss": values["loss_eval_results"][epoch],
            "Avg WER": wer,
            "Avg CER": cer,
        }
        print(values)
        self.tensorboard_writer.add_scalars(self.id, values, epoch)
        if self.log_params:
            for tag, value in parameters():
                tag = tag.replace(".", "/")
                self.tensorboard_writer.add_histogram(tag, to_np(value), epoch)
                self.tensorboard_writer.add_histogram(
                    tag + "/grad", to_np(value.grad), epoch
                )

    def load_previous_values(self, start_epoch, values):
        loss_results = values["loss_results"][:start_epoch]
        wer_results = values["wer_results"][:start_epoch]
        cer_results = values["cer_results"][:start_epoch]

        for i in range(start_epoch):
            values = {
                "Avg Train Loss": loss_results[i],
                "Avg WER": wer_results[i],
                "Avg CER": cer_results[i],
            }
            self.tensorboard_writer.add_scalars(self.id, values, i + 1)
