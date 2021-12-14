import time
import torch
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top1_16 = AverageMeter("Acc16@1", ":6.2f")
    top1_32 = AverageMeter("Acc32@1", ":6.2f")
    top1_64 = AverageMeter("Acc64@1", ":6.2f")
    top1_128 = AverageMeter("Acc128@1", ":6.2f")
    top1_256 = AverageMeter("Acc256@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, top1_16, top1_32, top1_64, top1_128, top1_256],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output512, output16, output32, output64, output128, output256 = model(images)

        loss = criterion(output512, target) + 10*criterion(output16, target) + 7*criterion(output32, target) + 5*criterion(output64, target) + 3*criterion(output128, target) + 2*criterion(output256, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output512, target, topk=(1, 5))
        acc1_16, _ = accuracy(output16, target, topk=(1, 5))
        acc1_32, _ = accuracy(output32, target, topk=(1, 5))
        acc1_64, _ = accuracy(output64, target, topk=(1, 5))
        acc1_128, _ = accuracy(output128, target, topk=(1, 5))
        acc1_256, _ = accuracy(output256, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top1_16.update(acc1_16.item(), images.size(0))
        top1_32.update(acc1_32.item(), images.size(0))
        top1_64.update(acc1_64.item(), images.size(0))
        top1_128.update(acc1_128.item(), images.size(0))
        top1_256.update(acc1_256.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)
    
    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch, save_to=None):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top1_16 = AverageMeter("Acc16@1", ":6.2f", write_val=False)
    top1_32 = AverageMeter("Acc32@1", ":6.2f", write_val=False)
    top1_64 = AverageMeter("Acc64@1", ":6.2f", write_val=False)
    top1_128 = AverageMeter("Acc128@1", ":6.2f", write_val=False)
    top1_256 = AverageMeter("Acc256@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5, top1_16, top1_32, top1_64, top1_128, top1_256], prefix="Test: "
    )
    batches_per_shard = 32
    rep_dict = None


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (images, target, paths) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output512, output16, output32, output64, output128, output256, rep = model(images, get_rep=True) # TODO: Note that output512 is 2048 dim probdistr

            if args.save_reps:
                if i % batches_per_shard == 0:
                    if rep_dict is not None:
                        path = args.save_reps + "/" + save_to + "_" + str(i) + ".pt"
                        print("Saving val_reps of shape", rep_dict["rep"].shape, "with labels", rep_dict["class"].shape, "to", path)
                        torch.save(rep_dict, path)

                    rep_dict = {"rep": rep,
                                "class": target,
                                "paths": paths}
                else:
                    rep_dict["rep"] = torch.cat((rep_dict["rep"], rep), dim=0)
                    rep_dict["class"] = torch.cat((rep_dict["class"], target), dim=0)
                    rep_dict["paths"] = rep_dict["paths"] + paths

                
                if i == len(val_loader)-1:
                    path = args.save_reps + "/" + save_to + "_" + str(i) + ".pt"
                    print("Saving val_reps of shape", rep_dict["rep"].shape, "with labels", rep_dict["class"].shape, "to", path)
                    torch.save(rep_dict, path)
            
            

            loss = criterion(output512, target) + criterion(output16, target) + criterion(output32, target) + criterion(output64, target) + criterion(output128, target) + criterion(output256, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output512, target, topk=(1, 5))
            acc1_16, _ = accuracy(output16, target, topk=(1, 5))
            acc1_32, _ = accuracy(output32, target, topk=(1, 5))
            acc1_64, _ = accuracy(output64, target, topk=(1, 5))
            acc1_128, _ = accuracy(output128, target, topk=(1, 5))
            acc1_256, _ = accuracy(output256, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top1_16.update(acc1_16.item(), images.size(0))
            top1_32.update(acc1_32.item(), images.size(0))
            top1_64.update(acc1_64.item(), images.size(0))
            top1_128.update(acc1_128.item(), images.size(0))
            top1_256.update(acc1_256.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

def modifier(args, epoch, model):
    return
