import os
import time
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from enum import Enum, auto
from collections import namedtuple

import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from prune import MaskedConv2d


def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def topic_log(topic_str, length=100):
    num_of_dash = length - (len(topic_str) + 2)
    left_num_of_dash = num_of_dash // 2
    right_num_of_dash = num_of_dash - left_num_of_dash
    print('\n\n' + '-' * left_num_of_dash + " " + topic_str + " " + '-' * right_num_of_dash)


def save_checkpoint(state, file_path='checkpoint.pth.tar'):
    torch.save(state, file_path)


def load_checkpoint(model, file, args):
    if os.path.isfile(file):
        print(f"=> loading checkpoint '{file}'")
        checkpoint = torch.load(file)
        try:
            best_top1_acc = checkpoint['best_prec1']
        except KeyError:
            best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint '{args.evaluate}'")
    else:
        print(f"=> no checkpoint found at '{file}'")
        raise Exception
    return model, best_top1_acc


def save_masked_checkpoint(model, mode, best_top1_acc, epoch, args):
    save_file_path = os.path.join(args.save_dir, f'checkpoint_{mode}_{args.method_str}_{epoch}.tar')
    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_top1_acc': best_top1_acc,
    }, file_path=save_file_path)
    return model


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-' * 70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def be_ignored(layer_name, model_mode):
    if (model_mode == 'd' and 'conv' in layer_name or
            model_mode == 'c' and 'fc' in layer_name or
            'mask' in layer_name or
            'bias' in layer_name or
            'bn' in layer_name):
        return True
    else:
        return False


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
        log(log_file,
            f'{name:25} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) '
            f'| total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(f'{name:25} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%)'
              f' | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    log(log_file, f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : '
                  f'{total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : '
          f'{total/nonzero:10.2f}x ({100 * (total-nonzero) / total:6.2f}% pruned)')


def initial_train(model, args, train_loader, val_loader, tok):
    cudnn.benchmark = True if args.use_cuda else False
    if tok == 'initial_train':
        epochs = args.epochs
        lr = args.train_lr
    else:  # for prune retrain
        epochs = args.prune_retrain_epochs
        lr = args.prune_retrain_lr
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    print(f'start epoch {args.start_epoch} / end epoch {epochs}')
    for epoch in range(args.start_epoch, epochs):
        print(f'\nin epoch {epoch}')
        optimizer = adjust_learning_rate(optimizer, epoch, args, lr)
        train(train_loader, model, optimizer, epoch, args, tok)  # train for one epoch
        top1_acc, top5_acc = validate(val_loader, model, args, topk=(1, 5))  # evaluate on validation set

        # record best top1_acc and save checkpoint
        if args.best_top1_acc < top1_acc:
            model = save_masked_checkpoint(model, tok, top1_acc, epoch, args)
            log(args.log_file_path, f"[epoch {epoch}]")
            log(args.log_file_path, f"initial_accuracy\t{top1_acc} ({top5_acc})")
            args.best_top1_acc = top1_acc

        #  if prune mode is "filter-gm" and during initial_train, then soft prune
        if args.model_mode != 'd' and args.prune_mode == 'filter-gm' and tok == "initial":
            if epoch % args.prune_interval == 0 or epoch == epoch - 1:
                prune_rates = model.get_conv_actual_prune_rates(args.prune_rates)
                model.prune_step(prune_rates, mode=args.prune_mode)

    model = save_masked_checkpoint(model, tok, args.best_top1_acc, epochs, args)
    return model


def train(train_loader, model, optimizer, epoch, args, tok=""):
    """ Train one epoch. """
    batch_times = AverageMeter()
    data_times = AverageMeter()
    losses = AverageMeter()
    top1s = AverageMeter()
    all_penalties = AverageMeter()
    fc_penalties = AverageMeter()
    conv_penalties = AverageMeter()
    criterion = nn.CrossEntropyLoss().to(args.device)
    penalty = nn.MSELoss(reduction='sum').to(args.device)
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_times.update(time.time() - end)  # measure data loading time

        target = target.to(args.device)
        input_var = Variable(input).to(args.device)
        target_var = Variable(target).to(args.device)

        # compute output and loss
        output = model(input_var)
        all_penalty, fc_penalty_, conv_penalty_ = get_layers_penalty(model, penalty, args, tok)
        loss = criterion(output, target_var) + all_penalty
        all_penalties.update(all_penalty)
        fc_penalties.update(fc_penalty_)
        conv_penalties.update(conv_penalty_)

        # compute gradient and update
        optimizer.zero_grad()
        loss.backward()
        if tok == "prune_retrain":  # let the gradient of the pruned node become 0
            for name, p in model.named_parameters():
                if be_ignored(name, args.model_mode):
                    continue
                tensor_arr = p.data.cpu().numpy()
                grad_tensor_arr = p.grad.data.cpu().numpy()
                grad_tensor_arr = np.where(tensor_arr == 0, 0, grad_tensor_arr)
                p.grad.data = torch.from_numpy(grad_tensor_arr).to(args.device)
        optimizer.step()
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        top1_acc = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1s.update(top1_acc, input.size(0))

        # measure elapsed time
        batch_times.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  f'Data {data_times.val:.3f} ({data_times.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'Prec {top1s.val:.3f} ({top1s.avg:.3f})\t'
                  f'Layer penalty {all_penalties.avg:.3f} ( fc: {fc_penalties.avg:.3f} , conv: {conv_penalties.avg:.3f} )')


def validate(val_loader, model, args, topk=(1,), tok=''):
    batch_times = AverageMeter()
    losses = AverageMeter()
    topk_objs = [AverageMeter() for _ in range(len(topk))]
    criterion = nn.CrossEntropyLoss().to(args.device)
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(args.device)
        input_var = Variable(input).to(args.device)
        target_var = Variable(target).to(args.device)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec_k = accuracy(output.data, target, topk=topk)
        losses.update(loss.item(), input.size(0))
        prec_str = str()
        for j in range(len(topk)):
            topk_objs[j].update(prec_k[j], input.size(0))
            prec_str += f'Prec {topk[j]}: {topk_objs[j].val:.3f} ({topk_objs[j].avg:.3f})\t'

        # measure elapsed time
        batch_times.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'{prec_str}')

    topk_list = [topk_obj.avg for topk_obj in topk_objs]
    prec_str = "  ".join([f'Prec {topk[i]} ({topk_list[i]:.3f})' for i in range(len(topk))])
    print(f' * {prec_str}\n')
    return topk_list


def quantized_retrain(model, args, layerName2quanIndices, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss().to(args.device)
    lr = args.quantize_retrain_lr
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    for epoch in range(args.qauntize_epochs):
        batch_times = AverageMeter()
        data_times = AverageMeter()
        losses = AverageMeter()
        top1s = AverageMeter()
        model.train()
        end = time.time()
        optimizer = adjust_learning_rate(optimizer, epoch, args, lr)

        # train for one epoch
        for i, (input, target) in enumerate(train_loader):
            data_times.update(time.time() - end)
            target = target.to(args.device)
            input_var = Variable(input).to(args.device)
            target_var = Variable(target).to(args.device)

            # compute output and update, and let nodes quantized in the same label have the same average gradient
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()
            for name, p in model.named_parameters():
                if be_ignored(name, args.model_mode):
                    continue
                quan_bits = 2 ** int(args.bits['fc' if 'fc' in name else 'conv'])
                quan_indices = layerName2quanIndices[name]
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_centers = list()
                for j in range(quan_bits):
                    grad_by_index = grad_tensor[quan_indices == j]
                    grad_center = np.mean(grad_by_index)
                    grad_centers.append(grad_center)
                grad_centers = np.array(grad_centers)
                grad_tensor = grad_centers[quan_indices]
                grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(args.device)
            optimizer.step()
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            top1_acc = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1s.update(top1_acc, input.size(0))

            # measure elapsed time
            batch_times.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                      f'Data {data_times.val:.3f} ({data_times.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec {top1s.val:.3f} ({top1s.avg:.3f})')

        # evaluate on validation set
        top1_acc, top5_acc = validate(val_loader, model, args, topk=(1, 5))
        log(args.log_file_path, f"[epoch {epoch}]")
        log(args.log_file_path, f"initial_accuracy\t{top1_acc}")
        log(args.log_file_path, f"initial_top5_accuracy\t{top5_acc}")
    return model


def show_quantized_channel2ds(model):
    conv_layer_index = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            conv4d = module.weight.data
            # Plot the channels of a convolution layer
            plt.figure(figsize=(50, 50))
            for k in range(5):
                plt.axis("off")
                plt.title(f"Conv layer {conv_layer_index}-{k}")
                images = np.transpose(
                    vutils.make_grid(
                        torch.unsqueeze(conv4d[k, :, :, :][:64], dim=1), padding=5, normalize=True
                    ).cpu(), (1, 2, 0)
                )
                plt.imshow(images)
                plt.show()
            conv_layer_index += 1


def conv_filter3d_delta_penalty(model, device, penalty, mode):
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_layer_weights(module.weight, model, name)
            penalty_filters = 0
            for i in range(left_conv_weights.shape[0] - 1):
                if 'filter' in mode:
                    penalty_filters += penalty(
                        left_conv_weights[i, :, :, :],
                        left_conv_weights[i+1, :, :, :]
                    )
                elif 'channel' in mode:
                    penalty_channels = 0
                    for j in range(left_conv_weights.shape[1] - 1):
                        penalty_channels += penalty(
                            left_conv_weights[i, j, :, :],
                            left_conv_weights[i, j+1, :, :]
                        )
                    penalty_filters += penalty_channels
            penalty_filters /= (left_conv_weights.shape[0] - 1)
            penalty_layers.append(penalty_filters)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def conv_part_filter3d_delta_penalty(model, device, penalty, mode):
    penalty_layers = list()
    part_filters_nums = 10  # hyperparameter
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_layer_weights(module.weight, model, name)

            # Sorted by absolute sum
            sum_of_filters = np.sum(
                np.abs(left_conv_weights.data.cpu().numpy().reshape(left_conv_weights.shape[0], -1)), 1
            )
            part_filters_indices = np.argsort(sum_of_filters)[:part_filters_nums]
            part_conv_weights = left_conv_weights[part_filters_indices, :, :, :]

            # Compute loss by part convolution weights
            penalty_filters = 0
            for i in range(part_conv_weights.shape[0] - 1):
                penalty_filters += penalty(
                    part_conv_weights[i, :, :, :],
                    part_conv_weights[i+1, :, :, :]
                )
            penalty_filters /= (part_conv_weights.shape[0] - 1)
            penalty_layers.append(penalty_filters)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def conv_group_filter3d_delta_penalty(model, device, penalty, mode):
    penalty_layers = list()
    filters_nums_per_group = 5  # hyperparameter
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_layer_weights(module.weight, model, name)

            # # Sorted by absolute sum and compute loss by sum of each group of convolution weights
            # sum_of_filters = np.sum(
            #     np.abs(left_conv_weights.data.cpu().numpy().reshape(left_conv_weights.shape[0], -1)), 1
            # )
            # sorted_filters_indices = np.argsort(sum_of_filters)

            # Sorted by ShortestPathFinder
            shortest_path_finder = ShortestPathFinder(left_conv_weights.data.cpu().numpy())
            sorted_filters_indices = shortest_path_finder.search(method_str='random', ).nodes_id
            print(sorted_filters_indices)

            penalty_filters = 0
            num_of_groups = 0
            for i in range(0, len(sorted_filters_indices), filters_nums_per_group):
                num_of_groups += 1
                group_filters_indices = sorted_filters_indices[i:i+filters_nums_per_group]
                group_conv_weights = left_conv_weights[group_filters_indices, :, :, :]
                if group_conv_weights.shape[0] > 1:  # Need at least two filter to compute loss
                    for j in range(group_conv_weights.shape[0] - 1):
                        penalty_filters += penalty(
                            group_conv_weights[j, :, :, :],
                            group_conv_weights[j+1, :, :, :]
                        )
            penalty_filters /= (left_conv_weights.shape[0] - num_of_groups)
            penalty_layers.append(penalty_filters)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def conv_position_mean_penalty(model, device, penalty, mode):
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_layer_weights(module.weight, model, name)
            penalty_pos = 0
            for ch in range(left_conv_weights.shape[1]):
                for i in range(left_conv_weights.shape[2]):
                    for j in range(left_conv_weights.shape[3]):
                        same_pos_weights = left_conv_weights[:, ch, i, j]
                        weights_mean = torch.mean(same_pos_weights)
                        penalty_pos += penalty(same_pos_weights, weights_mean)
            penalty_layers.append(penalty_pos)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def conv_matrix2d_mean_penalty(model, device, penalty, mode):
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_layer_weights(module.weight, model, name)
            penalty_matrix2d = 0
            for kn in range(left_conv_weights.shape[0]):
                for ch in range(left_conv_weights.shape[1]):
                    matrix2d_weights = left_conv_weights[kn, ch, :, :]
                    weights_mean = torch.mean(matrix2d_weights)
                    penalty_matrix2d += penalty(matrix2d_weights, weights_mean)
            penalty_layers.append(penalty_matrix2d)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def conv_width1d_delta_penalty(model, device, penalty, mode):
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_layer_weights(module.weight, model, name)
            width1ds_tensor = left_conv_weights.reshape(-1, left_conv_weights.shape[3])
            penalty_width1d = penalty(width1ds_tensor[:, :-1], width1ds_tensor[:, 1:])
            penalty_layers.append(penalty_width1d)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def block_fc_penalty(model, device, penalty):
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            penalty_blocks = 0
            fc_data = module.weight.data.cpu().numpy()
            num_of_partitions = int(model.partition_size[name])
            block_rows = fc_data.shape[0] // num_of_partitions
            block_cols = fc_data.shape[1] // num_of_partitions
            for i in range(num_of_partitions - 1):
                cur_block = module.weight[
                    i * block_rows: (i+1) * block_rows,
                    i * block_cols: (i+1) * block_cols,
                ]
                next_block = module.weight[
                    (i+1) * block_rows: (i+2) * block_rows,
                    (i+1) * block_cols: (i+2) * block_cols,
                ]
                penalty_blocks += penalty(cur_block, next_block)
            penalty_blocks /= (num_of_partitions - 1)
            penalty_layers.append(penalty_blocks)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def get_layers_penalty(model, penalty, args, tok):
    all_penalty = fc_penalty = conv_penalty = 0.0
    if args.model_mode != 'c' and args.alpha != 0:
        fc_penalty = block_fc_penalty(model, args.device, penalty)
        all_penalty += args.alpha * fc_penalty
    if args.model_mode != 'd' and tok == "prune_retrain" and args.beta != 0:
        if not ('filter' in args.prune_mode or 'channel' in args.prune_mode):
            conv_penalty = 0.0
        elif args.conv_loss_func == 'filter3d-delta':
            conv_penalty = conv_filter3d_delta_penalty(model, args.device, penalty, args.prune_mode)
        elif args.conv_loss_func == 'width1d-delta':
            conv_penalty = conv_width1d_delta_penalty(model, args.device, penalty, args.prune_mode)
        elif args.conv_loss_func == 'position-mean':
            conv_penalty = conv_position_mean_penalty(model, args.device, penalty, args.prune_mode)
        elif args.conv_loss_func == 'matrix2d-mean':
            conv_penalty = conv_matrix2d_mean_penalty(model, args.device, penalty, args.prune_mode)
        elif args.conv_loss_func == 'part-filter3d-delta':
            conv_penalty = conv_part_filter3d_delta_penalty(model, args.device, penalty, args.prune_mode)
        elif args.conv_loss_func == 'group-filter3d-delta':
            conv_penalty = conv_group_filter3d_delta_penalty(model, args.device, penalty, args.prune_mode)
        else:
            raise Exception
        all_penalty += args.beta * conv_penalty
    return all_penalty, fc_penalty, conv_penalty


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # pred shape: (batch_size, k)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = list()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_method_str(args):
    method_list = list()
    if args.model_mode != 'c':
        method_list.append(f'alpha_{args.alpha}')
    if args.model_mode != 'd':
        method_list.append(f'beta_{args.beta}')
    method_str = '_'.join(method_list)
    return method_str


def get_part_model_original_bytes(model, args):
    model_bytes = 0
    for name, param in model.named_parameters():
        if 'mask' in name or 'bn' in name:
            continue
        if 'conv' in name and args.model_mode != 'd' or 'fc' in name and args.model_mode != 'c' or 'bias' in name:
            model_bytes += param.data.cpu().numpy().nbytes
    return model_bytes


def get_unpruned_conv_layer_weights(layer_weights, model, name):
    if model.convLayerName2leftIndices is None:
        model.set_conv_indices_dict()
    left_filter_indices, left_channel_indices = model.convLayerName2leftIndices[name]
    unpruned_weights = layer_weights[left_filter_indices, :, :, :][:, left_channel_indices, :, :]
    return unpruned_weights


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
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


def adjust_learning_rate(optimizer, epoch, args, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adj_lr = lr * (0.1 ** (epoch // args.lr_drop_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = adj_lr
    return optimizer


class ShortestPathFinder:
    def __init__(self, nodes, max_iter_min=2.0):
        self._init()
        self.nodes = nodes
        self.max_iter_time = max_iter_min * 60

    def _init(self):
        self.shortest_selected_nodes_id = list()
        self.shortest_dist = -1
        self.global_step = None
        self.s_time = time.time()

    def compute_path_distance(self, nodes_id):
        sorted_nodes = self.nodes[nodes_id]
        return np.sum(np.power(sorted_nodes[1:] - sorted_nodes[:-1], 2))

    def search(self, method_str='random'):
        self._init()
        method = self.Method[method_str.upper()]
        if method is self.Method.RANDOM:
            self._find_approximate_shortest_path_by_random_search()
        elif method is self.Method.OPTIMIZE:
            self._find_optimized_shortest_path(set(range(len(self.nodes))), list())
        else:
            raise KeyError(method_str)
        return self._get_output_tuple()

    def _get_output_tuple(self):
        output = namedtuple("output", ["dist", "nodes_id"])
        return output(self.shortest_dist, self.shortest_selected_nodes_id)

    def _find_optimized_shortest_path(self, left_nodes_id, selected_nodes_id):
        if len(left_nodes_id) == 0:
            path_dist = self.compute_path_distance(selected_nodes_id)
            # print(selected_nodes_id, path_dist, self.shortest_dist)
            if len(self.shortest_selected_nodes_id) == 0 or path_dist < self.shortest_dist:
                self.shortest_selected_nodes_id = selected_nodes_id
                self.shortest_dist = path_dist
        else:
            for _id in left_nodes_id:
                _left_nodes_id = left_nodes_id.copy()
                _left_nodes_id.remove(_id)
                _selected_nodes_id = selected_nodes_id + [_id]
                self._find_optimized_shortest_path(_left_nodes_id, _selected_nodes_id)

    def _find_approximate_shortest_path_by_random_search(self):
        node_ids = list(range(len(self.nodes)))
        while True:
            shuffle(node_ids)
            path_dist = self.compute_path_distance(node_ids)
            if len(self.shortest_selected_nodes_id) == 0 or path_dist < self.shortest_dist:
                self.shortest_selected_nodes_id = node_ids
                self.shortest_dist = path_dist
            if time.time() - self.s_time >= self.max_iter_time:
                break

    class Method(Enum):
        OPTIMIZE = auto()
        RANDOM = auto()
