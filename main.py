#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import numpy as np
import model_builder
from torch.utils.data import DataLoader
from metrics import compute_metrics, print_computed_metrics, AverageMeter

from loss import FCE_loss
import pickle

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch_val', default=128, type=int,
                    metavar='N',
                    help='')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--moco-dim', default=512, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=2048, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# data embedding
parser.add_argument('--videodim', default=2048, type=int, help='')
parser.add_argument('--imgdim', default=2048, type=int, help='')
parser.add_argument('--textdim', default=768, type=int, help='')


# loss weight
parser.add_argument('--alpha', default=0.8, type=float, help='')
parser.add_argument('--beta', default=0.1, type=float, help='')
parser.add_argument('--gama', default=0.1, type=float, help='')
parser.add_argument('--margin', default=100.0, type=float, help='')

# dataset
parser.add_argument('--mvs', default=1, type=int,
                    help='use mvs dataset')
parser.add_argument('--mvslarge', default=0, type=int,
                    help='use mvslarge dataset')

if torch.cuda.is_available():
    # DEVICE = torch.device("cuda:" + str(config.gpu_name))
    DEVICE = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")
print("current deveice:", DEVICE)


def Eval_retrieval(model, eval_dataloader, dataset_name, epoch, args):
    model.eval()
    print('Evaluating retrieval on {} data'.format(dataset_name))
    with torch.no_grad():
        for i, data in enumerate(eval_dataloader):
            video = data['video'].cuda()
            videotext = data['videotext'].cuda()
            imgtext = data['imgtext'].cuda()
            img = data['img'].cuda()
            queue_index = data['queue_index'].cuda()
            first_cluster = data['first_cluster'].cuda()
            second_cluster = data['second_cluster'].cuda()
            third_cluster = data['third_cluster'].cuda()

            video = video.cuda(DEVICE, non_blocking=True)
            image = img.cuda(DEVICE, non_blocking=True)

            video = video.view(-1, video.shape[-1])

            vq, pq, vk, pk = model(video=video, img = image, videotext=videotext, imgtext=imgtext,queue_index=queue_index,first_cluster=first_cluster,
                                   second_cluster=second_cluster, third_cluster=third_cluster,args=args, tag='test')
            vq = torch.squeeze(vq)
            vk = torch.squeeze(vk)

            m = torch.matmul(vq, pq.t()).cpu().detach().numpy()
            metrics = compute_metrics(m)
            print_computed_metrics(0, dataset_name, metrics, epoch, args.batch_size, args.lr, args.epochs, args.alpha, args.beta, args.gama, args.moco_k,args.moco_dim, args.margin)

            #m = torch.matmul(pq, vk.t()).cpu().detach().numpy()
            metrics = compute_metrics(m, True)
            print_computed_metrics(1, dataset_name, metrics, epoch, args.batch_size, args.lr, args.epochs, args.alpha, args.beta, args.gama, args.moco_k,args.moco_dim, args.margin)

            return metrics['MR']

def main():

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = model_builder.Net(
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    print(model)

    model = model.to(DEVICE)
    # define loss function (criterion) and optimizer
    criterion1 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion2 = FCE_loss().cuda(args.gpu)

    """optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # load dataset
    if (args.mvs == 1):
        train_dataset = open("./dataset/MVS_train_dataset.pkl", 'rb')
        train_dataset = pickle.load(train_dataset)
        val_dataset = open("./dataset/MVS_val_dataset.pkl", 'rb')
        val_dataset = pickle.load(val_dataset)
        test_dataset = open("./dataset/MVS_test_dataset.pkl", 'rb')
        test_dataset = pickle.load(test_dataset)
    else:
        train_dataset = open("./dataset/MVS_large_train_dataset.pkl", 'rb')
        train_dataset = pickle.load(train_dataset)
        val_dataset = open("./dataset/MVS_large_val_dataset.pkl", 'rb')
        val_dataset = pickle.load(val_dataset)
        test_dataset = open("./dataset/MVS_large_test_dataset.pkl", 'rb')
        test_dataset = pickle.load(test_dataset)

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        batch_sampler=None,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
    )
    print(len(train_dataset))
    print(len(test_dataset))

    least_mr = 100

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion1, criterion2, optimizer, epoch, args)

        mr = Eval_retrieval(model, val_dataloader, 'val', epoch, args)

        if(mr < least_mr):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=True, filename='./checkpoint/checkpoint_{:04d}.pth.tar'.format(epoch))
            least_mr = mr

            Eval_retrieval(model, test_dataloader, 'test', epoch, args)


def train(train_loader, model, criterion1, criterion2, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()


    for i, data in enumerate(train_loader):
        video = data['video'].cuda()
        videotext = data['videotext'].cuda()
        imgtext = data['imgtext'].cuda()
        img = data['img'].cuda()
        queue_index = data['queue_index'].cuda()
        first_cluster = data['first_cluster'].cuda()
        second_cluster = data['second_cluster'].cuda()
        third_cluster = data['third_cluster'].cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:

            video = video.cuda(args.gpu, non_blocking=True)
            image = img.cuda(args.gpu, non_blocking=True)

        video = video.view(-1, video.shape[-1])

        output1, target1, output2, target2, videof, imgf, vtextf, ptextf = model(video=video, img=image, videotext=videotext, imgtext=imgtext,queue_index=queue_index,
                                                                           first_cluster=first_cluster, second_cluster=second_cluster, third_cluster=third_cluster, args=args)

        loss1 = criterion1(output1, target1) + criterion1(output2, target2)

        loss2 = criterion2(videof, imgf, args.margin) + criterion2(vtextf, ptextf, args.margin)

        loss3 = criterion2(videof, vtextf, args.margin) + criterion2(imgf, ptextf, args.margin)

        loss = float(args.alpha) * loss1 + float(args.beta) * loss2 + float(args.gama) * loss3

        acc1, acc5 = accuracy(output1, target1, topk=(1, 5))

        losses.update(loss.item(), video.size(0))

        top1.update(acc1[0], video.size(0))

        top5.update(acc5[0], video.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
