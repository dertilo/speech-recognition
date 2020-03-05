import argparse
import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from apex.parallel import DistributedDataParallel

from data_related.building_vocabulary import BLANK_CHAR
from warpctc_pytorch import CTCLoss

from data_related.data_loader import (
    AudioDataLoader,
    SpectrogramDataset,
    BucketingSampler,
    DistributedBucketingSampler,
)
from data_related.data_utils import read_jsonl
from decoder import GreedyDecoder
from logger import VisdomLogger, TensorBoardLogger
from model import DeepSpeech, supported_rnns
from test import evaluate
from train_util import train_one_epoch

parser = argparse.ArgumentParser(description="DeepSpeech training")
parser.add_argument(
    "--train-manifest",
    metavar="DIR",
    help="path to train manifest csv",
    default="spanish_train_manifest.csv",
)
parser.add_argument(
    "--val-manifest",
    metavar="DIR",
    help="path to validation manifest csv",
    default="spanish_eval_manifest.csv",
)
parser.add_argument("--sample-rate", default=16000, type=int, help="Sample rate")
parser.add_argument(
    "--batch-size", default=64, type=int, help="Batch size for training"
)
parser.add_argument(
    "--num-workers", default=16, type=int, help="Number of workers used in data-loading"
)
parser.add_argument(
    "--labels-path",
    default="spanish_vocab.json",
    help="Contains all characters for transcription",
)
parser.add_argument(
    "--window-size",
    default=0.02,
    type=float,
    help="Window size for spectrogram in seconds",
)
parser.add_argument(
    "--window-stride",
    default=0.01,
    type=float,
    help="Window stride for spectrogram in seconds",
)
parser.add_argument(
    "--window", default="hamming", help="Window type for spectrogram generation"
)
parser.add_argument("--hidden-size", default=1024, type=int, help="Hidden size of RNNs")
parser.add_argument("--hidden-layers", default=5, type=int, help="Number of RNN layers")
parser.add_argument(
    "--rnn-type", default="lstm", help="Type of the RNN. rnn|gru|lstm are supported"
)
parser.add_argument("--epochs", default=70, type=int, help="Number of training epochs")
parser.add_argument(
    "--cuda",
    dest="cuda",
    default=True,
    action="store_true",
    help="Use cuda to train model",
)
parser.add_argument(
    "--lr", "--learning-rate", default=3e-4, type=float, help="initial learning rate"
)
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument(
    "--max-norm",
    default=400,
    type=int,
    help="Norm cutoff to prevent explosion of gradients",
)
parser.add_argument(
    "--learning-anneal",
    default=1.01,
    type=float,
    help="Annealing applied to learning rate every epoch",
)
parser.add_argument(
    "--silent",
    dest="silent",
    action="store_true",
    help="Turn off progress tracking per iteration",
)
parser.add_argument(
    "--checkpoint",
    dest="checkpoint",
    default=True,
    action="store_true",
    help="Enables checkpoint saving of model",
)
parser.add_argument(
    "--tensorboard",
    dest="tensorboard",
    default=True,
    action="store_true",
    help="Turn on tensorboard graphing",
)
parser.add_argument(
    "--log-dir", default="tensorboard_logdir", help="Location of tensorboard log"
)
parser.add_argument(
    "--log-params",
    dest="log_params",
    action="store_true",
    help="Log parameter values and gradients",
)
parser.add_argument(
    "--id", default="Deepspeech training", help="Identifier for visdom/tensorboard run"
)
parser.add_argument(
    "--save-folder", default="checkpoints", help="Location to save epoch models"
)
parser.add_argument(
    "--continue-from", default="", help="Continue from checkpoint model"
)
parser.add_argument(
    "--finetune",
    dest="finetune",
    action="store_true",
    help='Finetune the model from checkpoint "continue_from"',
)
parser.add_argument(
    "--speed-volume-perturb",
    default=False,
    dest="speed_volume_perturb",
    action="store_true",
    help="Use random tempo and gain perturbations.",
)
parser.add_argument(
    "--spec-augment",
    default=False,
    dest="spec_augment",
    action="store_true",
    help="Use simple spectral augmentation on mel spectograms.",
)

parser.add_argument(
    "--noise-dir",
    default=None,
    help="Directory to inject noise into audio. If default, noise Inject not added",
)
parser.add_argument(
    "--noise-prob", default=0.4, help="Probability of noise being added per sample"
)
parser.add_argument(
    "--noise-min",
    default=0.0,
    help="Minimum noise level to sample from. (1.0 means all noise, not original signal)",
    type=float,
)
parser.add_argument(
    "--noise-max",
    default=0.5,
    help="Maximum noise levels to sample from. Maximum 1.0",
    type=float,
)
parser.add_argument(
    "--feature-type", default="stft", choices=["stft", "mfcc", "mel"], type=str,
)
parser.add_argument(
    "--no-shuffle",
    dest="no_shuffle",
    action="store_true",
    help="Turn off shuffling and sample from dataset based on sequence length (smallest to largest)",
)
parser.add_argument(
    "--no-sortaGrad",
    default=True,
    dest="no_sorta_grad",
    action="store_true",
    help="Turn off ordering of dataset on sequence length for the first epoch.",
)
parser.add_argument(
    "--no-bidirectional",
    dest="bidirectional",
    action="store_false",
    default=True,
    help="Turn off bi-directional RNNs, introduces lookahead convolution",
)
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:1550",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument("--rank", default=0, type=int, help="The rank of this process")
parser.add_argument(
    "--gpu-rank",
    default=None,
    help="If using distributed parallel for multi-gpu, sets the GPU for the process",
)
parser.add_argument("--seed", default=123456, type=int, help="Seed to generators")
parser.add_argument("--opt-level", default="O1", type=str)
parser.add_argument("--keep-batchnorm-fp32", type=str, default=None)
parser.add_argument("--loss-scale", default=1, type=str)

torch.manual_seed(123456)
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


if __name__ == "__main__":
    args = parser.parse_args()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
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
        if "spanish" in args.labels_path:
            labels = str("".join(next(iter(read_jsonl(args.labels_path)))))
        else:
            labels = [
                "_",
                "'",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                " ",
            ]

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
        speed_volume_perturb=args.speed_volume_perturb,
        spec_augment=args.spec_augment,
    )
    test_dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=args.val_manifest,
        labels=labels,
        normalize=True,
        speed_volume_perturb=False,
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
    print(model)
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
