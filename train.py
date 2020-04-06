import argparse
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from apex.parallel import DistributedDataParallel
from util import data_io

from data_related.audio_feature_extraction import AudioFeaturesConfig
from warpctc_pytorch import CTCLoss

from data_related.char_stt_dataset import DataConfig, CharSTTDataset
from data_related.data_loader import (
    BucketingSampler,
    DistributedBucketingSampler,
    AudioDataLoader,
)
from decoder import GreedyDecoder
from logger import TensorBoardLogger
from model import DeepSpeech, supported_rnns
from test import evaluate
from train_util import train_one_epoch
from utils import USE_GPU, BLANK_SYMBOL, SPACE, HOME

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


def build_datasets():
    # fmt: off
    labels = [BLANK_SYMBOL, "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", SPACE]
    # fmt: on
    from corpora.librispeech import librispeech_corpus

    HOME = os.environ["HOME"]
    asr_path = HOME + "/data/asr_data"
    raw_data_path = asr_path + "/ENGLISH/LibriSpeech"
    conf = DataConfig(labels)
    audio_conf = AudioFeaturesConfig()
    corpus = {
        k: v
        for folder in ["train-clean-100"]
        for k, v in librispeech_corpus(os.path.join(raw_data_path, folder)).items()
    }
    assert len(corpus) > 0
    train_dataset = CharSTTDataset(corpus, conf, audio_conf)
    audio_conf = AudioFeaturesConfig()
    corpus = {
        k: v
        for folder in ["dev-clean","dev-other"]
        for k, v in librispeech_corpus(os.path.join(raw_data_path, folder)).items()
    }
    eval_dataset = CharSTTDataset(corpus, conf, audio_conf)
    return train_dataset, eval_dataset


def set_seeds(seed):
    torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

#fmt: off
parser = argparse.ArgumentParser(description="multiproc_args")
parser.add_argument("--rank", default=0, type=int, help="The rank of this process")
parser.add_argument("--gpu-rank", default=None, help="If using distributed parallel for multi-gpu, sets the GPU for the process")
parser.add_argument("--world-size",type=int, default=0)
#fmt: on

if __name__ == "__main__":

    multiproc_args = parser.parse_args()
    args = argparse.Namespace(**data_io.read_json("train_config.json"))
    # Set seeds for determinism
    set_seeds(args.seed)

    world_size = multiproc_args.world_size
    is_distributed = world_size > 1
    multiproc_args.distributed=is_distributed
    main_proc = True
    device = torch.device("cuda" if USE_GPU else "cpu")
    rank = multiproc_args.rank
    if is_distributed:
        gpu_rank = multiproc_args.gpu_rank
        if gpu_rank:
            torch.cuda.set_device(int(gpu_rank))
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank,
        )
        main_proc = rank == 0  # Only the first proc should save models

    train_dataset, eval_dataset = build_datasets()

    save_folder = os.path.join(args.save_folder, args.id)
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists
    things_to_monitor = [
        "loss_results",
        "cer_results",
        "wer_results",
        "loss_eval_results",
    ]
    log_data = {k: torch.Tensor(args.epochs) for k in things_to_monitor}

    if main_proc:
        tensorboard_logdir = HOME+'/data/tensorboard_logs'
        tensorboard_logger = TensorBoardLogger(args.id, tensorboard_logdir)

    start_epoch, start_iter, optim_state, amp_state = 0, 0, None, None
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(
            args.continue_from, map_location=lambda storage, loc: storage
        )

        model = DeepSpeech.load_model_package(package)
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

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(
            rnn_hidden_size=args.hidden_size,
            nb_layers=args.hidden_layers,
            vocab_size=len(train_dataset.char2idx),
            rnn_type=supported_rnns[rnn_type],
            input_feature_dim=train_dataset.audio_fe.feature_dim,
            bidirectional=args.bidirectional,
        )
    if rank==0:
        print(model)

    decoder = GreedyDecoder(train_dataset.char2idx)
    if not is_distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(
            train_dataset,
            batch_size=args.batch_size,
            num_replicas=world_size,
            rank=rank,
        )
    train_loader = AudioDataLoader(
        train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler
    )
    test_loader = AudioDataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters, lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=1e-5
    )
    if USE_GPU and args.opt_level is not None:
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

    if is_distributed:
        model = DistributedDataParallel(model)
    # print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    criterion = CTCLoss(blank=train_dataset.char2idx[BLANK_SYMBOL])
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
            multiproc_args,
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
                args=multiproc_args,
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

        if main_proc:
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
