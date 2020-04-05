import time

import torch

from utils import calc_loss, check_loss, USE_GPU
from apex import amp
from apex.parallel import DistributedDataParallel


def train_one_epoch(
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
):
    end = time.time()
    avg_loss = 0
    for i, (data) in enumerate(train_loader, start=start_iter):
        if i == len(train_sampler):
            break
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.to(device)

        out, output_sizes = model(inputs, input_sizes)
        assert out.size(0) == inputs.size(0)
        loss, loss_value = calc_loss(
            out,
            output_sizes,
            criterion,
            targets,
            target_sizes,
            device,
            args.distributed,
            args.world_size,
        )

        # Check to ensure valid loss was calculated
        loss_is_valid, error = check_loss(loss, loss_value)
        if loss_is_valid:
            optimizer.zero_grad()
            # compute gradient
            if USE_GPU:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            optimizer.step()
        else:
            print(error)
            print("Skipping grad update")
            loss_value = 0

        avg_loss += loss_value

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.silent and i % 100 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss:.4f} ({loss:.4f})\t".format(
                    (epoch + 1),
                    (i + 1),
                    len(train_sampler),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_value,
                )
            )

    avg_loss /= i
    return avg_loss
