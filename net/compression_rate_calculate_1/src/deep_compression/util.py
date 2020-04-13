import argparse
import os
import time
import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)
def layer2torch(model, folder_path):
    for name, modules in model.named_children():
        torch.save(modules.weight.data,f"{folder_path}/{name}.txt")

def parameters2list(model):
    weight_list = []
    for module in model:

        dev = module.weight.device
        layer_weight = module.weight.data.cpu().numpy()
        shape = layer_weight.shape
        layer_weight = layer_weight.reshape(1, -1)
        for i in range(len(layer_weight)):
            mat = layer_weight[i].reshape(-1)
            if i == 0:
                weights = mat
            elif i == 1:
                weights = np.append([weights],[mat], axis=0)
            else:
                weights = np.append(weights,[mat], axis=0)
        weights = weights.reshape(shape)
        weight_list.append(weights)

    return weight_list
def save_parameters(file_folder, quantized_parameters):
    for i in range(len(quantized_parameters)):
        layer_weight = quantized_parameters[i]
        shape = layer_weight.shape
        if len(shape)>1:
            f = open(file_folder+'/layer_'+str(i)+'.txt','a')
            if len(shape)==2:
                for weight in layer_weight:
                    for j in range(len(weight)):
                        f.write(str(weight[j])+"\t")
                    f.write("\n")
            else:
                f.write(str(shape)+"\n")
                for filter_weight in layer_weight:
                    for weight in filter_weight:
                        weight = weight.reshape(-1)
                        for a_weight in weight:
                            f.write(str(a_weight)+"\t")
                        f.write("\n")
                    f.write("\n")

def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)

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
        log(log_file, f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    log(log_file, f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')

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
def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
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
    print(' * Prec {top1.avg:.3f}'.format(top1=top1))
    return top1.avg
