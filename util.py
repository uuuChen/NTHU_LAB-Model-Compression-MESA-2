# System
import os
import torch
import time
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

# Custom
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
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint '{args.evaluate}'")
    else:
        print(f"=> no checkpoint found at '{file}'")
        raise Exception
    return model, best_prec1


def save_masked_checkpoint(model, mode, best_prec1, epoch, args):
    save_file_path = os.path.join(args.save_dir, f'checkpoint_{mode}_{args.method_str}_{epoch}.tar')
    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
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
        log(log_file, f'{name:25} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(f'{name:25} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    log(log_file, f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x ({100 * (total-nonzero) / total:6.2f}% pruned)')


def initial_train(model, args, train_loader, val_loader, tok):
    cudnn.benchmark = True if args.use_cuda else False
    if tok == 'initial_train':
        epochs = args.epochs
        lr = args.train_lr
    else:  # for prune retrain
        epochs = args.prune_retrain_epochs
        lr = args.prune_retrain_lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print(f'start epoch {args.start_epoch} / end epoch {epochs}')
    for epoch in range(args.start_epoch, epochs):
        print(f'\nin epoch {epoch}')
        optimizer = adjust_learning_rate(optimizer, epoch, args, lr)
        train(train_loader, model, optimizer, epoch, args, tok)  # train for one epoch
        prec1, prec5 = validate(val_loader, model, args, topk=(1, 5))  # evaluate on validation set

        # record best prec1 and save checkpoint
        if args.best_prec1 < prec1 or tok == "prune_retrain":
            model = save_masked_checkpoint(model, tok, prec1, epoch, args)
            log(args.log_file_path, f"[epoch {epoch}]")
            log(args.log_file_path, f"initial_accuracy\t{prec1} ({prec5})")
            args.best_prec1 = prec1

        #  if prune mode is "filter-gm" and during initial_train, then soft prune
        if args.model_mode != 'd' and args.prune_mode == 'filter-gm' and tok == "initial":
            if epoch % args.prune_interval == 0 or epoch == epoch - 1:
                prune_rates = model.get_conv_actual_prune_rates(args.prune_rates)
                model.prune_step(prune_rates, mode=args.prune_mode)

    model = save_masked_checkpoint(model, tok, args.best_prec1, epochs, args)
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
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1s.update(prec1, input.size(0))

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
    topk_avg_meters = [AverageMeter() for _ in range(len(topk))]
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
        preck = accuracy(output.data, target, topk=topk)
        losses.update(loss.item(), input.size(0))
        prec_str = str()
        for j in range(len(topk)):
            topk_avg_meters[j].update(preck[j], input.size(0))
            prec_str += f'Prec {topk[j]}: {topk_avg_meters[j].val:.3f} ({topk_avg_meters[j].avg:.3f})\t'

        # measure elapsed time
        batch_times.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'{prec_str}')

    topk_prec_avg = [avg_meter.avg for avg_meter in topk_avg_meters]
    prec_str = "  ".join([f'Prec {topk[i]} ({topk_prec_avg[i]:.3f})' for i in range(len(topk))])
    print(f' * {prec_str}\n')
    return topk_prec_avg


def quantized_retrain(model, args, quan_name2labels, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss().to(args.device)
    lr = args.quantize_retrain_lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
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
                quan_labels = quan_name2labels[name]
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_centers = list()
                for j in range(quan_bits):
                    grad_by_index = grad_tensor[quan_labels == j]
                    grad_center = np.mean(grad_by_index)
                    grad_centers.append(grad_center)
                grad_centers = np.array(grad_centers)
                grad_tensor = grad_centers[quan_labels]
                grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(args.device)
            optimizer.step()
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1s.update(prec1, input.size(0))

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
        prec1, prec5 = validate(val_loader, model, args, topk=(1, 5))
        log(args.log_file_path, f"[epoch {epoch}]")
        log(args.log_file_path, f"initial_accuracy\t{prec1}")
        log(args.log_file_path, f"initial_top5_accuracy\t{prec5}")
    return model


def conv_delta_penalty(model, device, penalty, mode):
    if not ('filter' in mode or 'channel' in mode):
        return 0.0
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_weights(module.weight, model, name)
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


def conv_position_mean_penalty(model, device, penalty, mode):
    if not ('filter' in mode or 'channel' in mode):
        return 0.0
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_weights(module.weight, model, name)
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
    if not ('filter' in mode or 'channel' in mode):
        return 0.0
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_weights(module.weight, model, name)
            penalty_matrix2d = 0
            for kn in range(left_conv_weights.shape[0]):
                for ch in range(left_conv_weights.shape[1]):
                    matrix2d_weights = left_conv_weights[kn, ch, :, :]
                    weights_mean = torch.mean(matrix2d_weights)
                    penalty_matrix2d += penalty(matrix2d_weights, weights_mean)
            penalty_layers.append(penalty_matrix2d)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def conv_matrix1d_delta_penalty(model, device, penalty, mode):
    if not ('filter' in mode or 'channel' in mode):
        return 0.0
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            left_conv_weights = get_unpruned_conv_weights(module.weight, model, name)
            matrix1ds_tensor = left_conv_weights.reshape(-1, left_conv_weights.shape[3])
            penalty_matrix1d = penalty(matrix1ds_tensor[:, :-1], matrix1ds_tensor[:, 1:])
            penalty_layers.append(penalty_matrix1d)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def fc_penalty(model, device, penalty):
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            penalty_blocks = 0
            fc_data = module.weight.data.cpu().numpy()
            num_of_partizions = int(model.partition_size[name])
            for i in range(num_of_partizions - 1):
                block_rows = fc_data.shape[0] // num_of_partizions
                block_cols = fc_data.shape[1] // num_of_partizions
                cur_block = module.weight[
                    i * block_rows: (i+1) * block_rows,
                    i * block_cols: (i+1) * block_cols,
                ]
                next_block = module.weight[
                    (i+1) * block_rows: (i+2) * block_rows,
                    (i+1) * block_cols: (i+2) * block_cols,
                ]
                penalty_blocks += penalty(cur_block, next_block)
            penalty_blocks /= (num_of_partizions - 1)
            penalty_layers.append(penalty_blocks)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def get_layers_penalty(model, penalty, args, tok):
    all_penalty = fc_penalty_ = conv_penalty_ = 0.0
    if args.model_mode != 'c' and args.alpha != 0:
        fc_penalty_ = fc_penalty(model, args.device, penalty)
        all_penalty += args.alpha * fc_penalty_
    if args.model_mode != 'd' and tok == "prune_retrain" and args.beta != 0:
        if args.conv_loss_func == 'delta':
            conv_penalty_ = conv_delta_penalty(model, args.device, penalty, args.prune_mode)
        elif args.conv_loss_func == 'matrix1d-delta':
            conv_penalty_ = conv_matrix1d_delta_penalty(model, args.device, penalty, args.prune_mode)
        elif args.conv_loss_func == 'position-mean':
            conv_penalty_ = conv_position_mean_penalty(model, args.device, penalty, args.prune_mode)
        elif args.conv_loss_func == 'matrix2d-mean':
            conv_penalty_ = conv_matrix2d_mean_penalty(model, args.device, penalty, args.prune_mode)
        else:
            raise Exception
        all_penalty += args.beta * conv_penalty_
    return all_penalty, fc_penalty_, conv_penalty_


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = list()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
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
        if ('conv' in name and args.model_mode != 'd' or
                'fc' in name and args.model_mode != 'c' or
                'bias' in name):
            model_bytes += param.data.cpu().numpy().nbytes
    return model_bytes


def get_unpruned_conv_weights(conv_weights, model, name):
    if model.conv2leftIndicesDict is None:
        model.set_conv_prune_indices_dict()
    left_filter_indices, left_channel_indices = model.conv2leftIndicesDict[name]
    unpruned_conv_weights = conv_weights[left_filter_indices, :, :, :][:, left_channel_indices, :, :]
    return unpruned_conv_weights


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


def adjust_learning_rate(optimizer, epoch, args, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adj_lr = lr * (0.1 ** (epoch // args.lr_drop_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = adj_lr
    return optimizer
