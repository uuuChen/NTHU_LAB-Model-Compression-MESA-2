import os
import torch
import math
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
def load_checkpoint(model, file,args):
    if os.path.isfile(file):
        print ("=> loading checkpoint '{}'".format(file))
        checkpoint = torch.load(file)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        '''
        temp1=torch.index_select(model.fc1.weight, 0, model.rowp1.cuda())
        model.fc1.weight=torch.nn.Parameter(torch.index_select(temp1, 1, model.colp1.cuda()))
        temp2=torch.index_select(model.fc2.weight, 0, model.rowp2.cuda())
        model.fc2.weight=torch.nn.Parameter(torch.index_select(temp2, 1, model.colp2.cuda()))
        temp3=torch.index_select(model.fc3.weight, 0, model.rowp3.cuda())
        model.fc3.weight=torch.nn.Parameter(torch.index_select(temp3, 1, model.colp3.cuda()))
        print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        '''
    else:
        print ("=> no checkpoint found at '{}'".format(file))
    return model

def layer2torch(folder_path, model):
    for name, modules in model.named_children():
        torch.save(modules.weight.data,f"{folder_path}/{name}.txt")

def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)
def accuracy(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def validate(val_loader, model, alpha):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()

    penalty = nn.MSELoss(reduction='sum').cuda()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)+alpha*fc_penalty(model,penalty)
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    print(' * Prec {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg
def fc_penalty(model, penalty):
    penalty_fc1=penalty_fc2=penalty_fc3=0

    for i in range(int(model.partition_size['fc1'])-1):
        penalty_fc1 += penalty(model.fc1.weight[i*model.block_row_size1:(i+1)*model.block_row_size1:1,i*model.block_col_size1:(i+1)*model.block_col_size1:1],
            model.fc1.weight[(i+1)*model.block_row_size1:(i+2)*model.block_row_size1:1,(i+1)*model.block_col_size1:(i+2)*model.block_col_size1:1])
    for i in range(int(model.partition_size['fc2'])-1):

        penalty_fc2 += penalty(model.fc2.weight[i*model.block_row_size2:(i+1)*model.block_row_size2:1,i*model.block_col_size2:(i+1)*model.block_col_size2:1],
            model.fc2.weight[(i+1)*model.block_row_size2:(i+2)*model.block_row_size2:1,(i+1)*model.block_col_size2:(i+2)*model.block_col_size2:1])
    for i in range(int(model.partition_size['fc3'])-1):

        penalty_fc3 += penalty(model.fc3.weight[i*model.block_row_size3:(i+1)*model.block_row_size3:1,i*model.block_col_size3:(i+1)*model.block_col_size3:1],
            model.fc3.weight[(i+1)*model.block_row_size3:(i+2)*model.block_row_size3:1,(i+1)*model.block_col_size3:(i+2)*model.block_col_size3:1])
    
    penalty_fc1=penalty_fc1/(int(model.partition_size['fc1'])-1)
    penalty_fc2=penalty_fc2/(int(model.partition_size['fc2'])-1)
    penalty_fc3=penalty_fc3/(int(model.partition_size['fc3'])-1)
    penalty=0.33*(penalty_fc1+penalty_fc2+penalty_fc3)
    return penalty.cuda()

def print_nonzeros(model, log_file):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        log(log_file, f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%)'
                      f' | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | '
              f'total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    log(log_file, f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : '
                  f'{total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x '
          f' ({100 * (total-nonzero) / total:6.2f}% pruned)')


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