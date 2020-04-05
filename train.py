import argparse
import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
# from apex import amp
# from apex.parallel import DistributedDataParallel
from util import data_io

from data_related.building_vocabulary import BLANK_CHAR
# from warpctc_pytorch import CTCLoss

from data_related.data_loader import (
    BucketingSampler,
    DistributedBucketingSampler,
    AudioDataLoader)
from decoder import GreedyDecoder
from logger import TensorBoardLogger
from model import DeepSpeech, supported_rnns
from test import evaluate
from train_util import train_one_epoch
USE_GPU = torch.cuda.is_available()

torch.manual_seed(123456)
if USE_GPU:
    torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_labels(labels_path):
    if "spanish" in labels_path:
        labels = str("".join(next(iter(data_io.read_jsonl(labels_path)))))
    else:
        # fmt: off
        labels = ["_", "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                  " "]
        # fmt: on
    return labels


if __name__ == "__main__":
    args = argparse.Namespace(**data_io.read_json('train_config.json'))
    # Set seeds for determinism
    torch.manual_seed(args.seed)
    if USE_GPU:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.distributed = args.world_size > 1
    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        main_proc = args.rank == 0  # Only the first proc should save models
    save_folder = os.path.join(args.save_folder, args.id)
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists
    things_to_monitor = [
        "loss_results",
        "cer_results",
        "wer_results",
        "loss_eval_results",
    ]
    log_data = {k: torch.Tensor(args.epochs) for k in things_to_monitor}

    if main_proc and args.tensorboard:
        tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    start_epoch, start_iter, optim_state, amp_state = 0, 0, None, None
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(
            args.continue_from, map_location=lambda storage, loc: storage
        )
        if "feature_type" not in package["audio_conf"]:
            package["audio_conf"]["feature_type"] = "stft"

        model = DeepSpeech.load_model_package(package)
        labels = model.labels
        audio_conf = model.audio_conf
        if not args.finetune:  # Don't want to restart training
            optim_state = package["optim_dict"]
            amp_state = package["amp"]
            start_epoch = (
                int(package.get("epoch", 1)) - 1
            )  # Index start at 0 for training
            start_iter = package.get("iteration", None)
            if start_iter is None:
                start_epoch += (
                    1  # We saved model after epoch finished, start at the next epoch.
                )
                start_iter = 0
            else:
                start_iter += 1
            log_data = {k: package[k] for k in things_to_monitor}
            best_wer = log_data["wer_results"][start_epoch]
            # if main_proc and args.tensorboard:  # Previous scores to tensorboard logs #TODO: should not be necessary!
            #     tensorboard_logger.load_previous_values(start_epoch, package)
    else:
        labels = get_labels(args.labels_path)

        audio_conf = dict(
            sample_rate=args.sample_rate,
            window_size=args.window_size,
            window_stride=args.window_stride,
            window=args.window,
            noise_dir=args.noise_dir,
            noise_prob=args.noise_prob,
            noise_levels=(args.noise_min, args.noise_max),
            feature_type=args.feature_type,
        )

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(
            rnn_hidden_size=args.hidden_size,
            nb_layers=args.hidden_layers,
            labels=labels,
            rnn_type=supported_rnns[rnn_type],
            audio_conf=audio_conf,
            bidirectional=args.bidirectional,
        )

    decoder = GreedyDecoder(labels)
    train_dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=args.train_manifest,
        labels=labels,
        normalize=True,
        signal_augment=False,
        spec_augment=True,
    )
    test_dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=args.val_manifest,
        labels=labels,
        normalize=True,
        signal_augment=False,
        spec_augment=False,
    )
    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(
            train_dataset,
            batch_size=args.batch_size,
            num_replicas=args.world_size,
            rank=args.rank,
        )
    train_loader = AudioDataLoader(
        train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler
    )
    test_loader = AudioDataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters, lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=1e-5
    )
    if args.cuda and args.opt_level is not None:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=args.opt_level,
            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
            loss_scale=args.loss_scale,
        )

    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    if amp_state is not None:
        amp.load_state_dict(amp_state)

    if args.distributed:
        model = DistributedDataParallel(model)
    # print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    criterion = CTCLoss(blank=labels.index(BLANK_CHAR))
    batch_time = AverageMeter()
    data_time = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start_epoch_time = time.time()
        avg_loss = train_one_epoch(
            model,
            train_loader,
            start_iter,
            train_sampler,
            data_time,
            batch_time,
            criterion,
            args,
            optimizer,
            epoch,
            device,
        )

        epoch_time = time.time() - start_epoch_time
        print(
            "Training Summary Epoch: [{0}]\t"
            "Time taken (s): {epoch_time:.0f}\t"
            "Average Loss {loss:.3f}\t".format(
                epoch + 1, epoch_time=epoch_time, loss=avg_loss
            )
        )

        start_iter = 0  # Reset start iteration for next epoch
        with torch.no_grad():
            wer, cer, avg_val_loss, output_data = evaluate(
                test_loader=test_loader,
                device=device,
                model=model,
                decoder=decoder,
                target_decoder=decoder,
                criterion=criterion,
                args=args,
            )
        log_data["loss_results"][epoch] = avg_loss
        log_data["loss_eval_results"][epoch] = avg_val_loss
        log_data["wer_results"][epoch] = wer
        log_data["cer_results"][epoch] = cer
        print(
            "Validation Summary Epoch: [{0}]\t"
            "Average WER {wer:.3f}\t"
            "Average CER {cer:.3f}\t".format(epoch + 1, wer=wer, cer=cer)
        )

        if args.tensorboard and main_proc:
            tensorboard_logger.update(epoch, log_data, model.named_parameters())

        if main_proc and args.checkpoint:
            file_path = "%s/deepspeech_%d.pth.tar" % (save_folder, epoch + 1)
            torch.save(
                DeepSpeech.serialize(
                    model, optimizer=optimizer, amp=amp, epoch=epoch, log_data=log_data,
                ),
                file_path,
            )
        # anneal lr
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] / args.learning_anneal
        print("Learning rate annealed to: {lr:.6f}".format(lr=g["lr"]))

        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)
