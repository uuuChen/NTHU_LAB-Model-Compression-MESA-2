import os
import torch
import time
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from prune import MaskedConv2d


def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


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
        raise
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


def initial_train(model, args, train_loader, val_loader, tok):
    best_prec1 = 0.0
    cudnn.benchmark = True if args.use_cuda else False
    if tok == 'prune_retrain':
        epochs = args.reepochs
        args.lr = 1e-4
    else:
        epochs = args.epochs
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    print(f'start epoch {args.start_epoch} / end epoch {epochs}')
    for epoch in range(args.start_epoch, epochs):
        print(f'\nin epoch {epoch}')
        optimizer = adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, optimizer, epoch, args, tok)  # train for one epoch
        prec1, prec5 = validate(val_loader, model, args, topk=(1, 5))  # evaluate on validation set

        # record best prec1 and save checkpoint
        if best_prec1 < prec1 or tok == "prune_retrain":
            model = save_masked_checkpoint(model, tok, best_prec1, epoch, args)
            log(f"{args.save_dir}/{args.log}", f"[epoch {epoch}]")
            log(f"{args.save_dir}/{args.log}", f"initial_accuracy\t{prec1} ({prec5})")
            best_prec1 = prec1

        #  if prune mode is "filter-gm" and during initial_train, then soft prune
        if args.model_mode != 'd' and args.prune_mode == 'filter-gm' and tok == "initial":
            if epoch % args.prune_interval == 0 or epoch == epoch - 1:
                prune_rates = model.get_conv_actual_prune_rates(args.prune_rates)
                model.prune_step(prune_rates, mode=args.prune_mode)

    model = save_masked_checkpoint(model, tok, best_prec1, epochs, args)
    return model


def quantized_retrain(model, args, quantized_index_list, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss().to(args.device)
    args.lr = 1e-4
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(args.qauntize_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        model.train()
        end = time.time()
        optimizer = adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            target = target.to(args.device)
            input_var = torch.autograd.Variable(input).to(args.device)
            target_var = torch.autograd.Variable(target).to(args.device)

            # compute output and update
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()
            k = 0
            for name, p in model.named_parameters():
                if (args.model_mode == 'd' and 'conv' in name or
                        args.model_mode == 'c' and 'fc' in name or
                        'mask' in name or
                        'bias' in name or
                        'bn' in name):
                    continue
                quantized_index = quantized_index_list[k]
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_center_array = list()
                for j in range(2 ** int(args.bits['fc' if 'fc' in name else 'conv'])):
                    grad_by_index = grad_tensor[quantized_index == j]
                    grad_center = np.mean(grad_by_index)
                    grad_center_array.append(grad_center)
                grad_center_array = np.array(grad_center_array)
                grad_tensor = grad_center_array[quantized_index]
                grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(args.device)
                k += 1
            optimizer.step()
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec {top1.val:.3f} ({top1.avg:.3f})')

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, args, topk=(1, 5))
        model = save_masked_checkpoint(model, 'quantized_re', prec5, epoch, args)
        log(f"{args.save_dir}/{args.log}", f"[epoch {epoch}]")
        log(f"{args.save_dir}/{args.log}", f"initial_accuracy\t{prec1}")
        log(f"{args.save_dir}/{args.log}", f"initial_top5_accuracy\t{prec5}")

    return model


def validate(val_loader, model, args, topk=(1,), tok=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    topk_avg_meters = [AverageMeter() for _ in range(len(topk))]

    criterion = nn.CrossEntropyLoss().to(args.device)
    penalty = nn.MSELoss(reduction='sum').to(args.device)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(args.device)
        input_var = torch.autograd.Variable(input).to(args.device)
        target_var = torch.autograd.Variable(target).to(args.device)

        # compute output
        output = model(input_var)
        all_penalty, fc_penalty_, conv_penalty_ = get_layers_penalty(model, penalty, args, tok)
        loss = criterion(output, target_var) + all_penalty
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        preck = accuracy(output.data, target, topk=topk)
        losses.update(loss.item(), input.size(0))
        prec_str = str()
        for j in range(len(topk)):
            topk_avg_meters[j].update(preck[j], input.size(0))
            prec_str += f'Prec {topk[j]} {topk_avg_meters[j].val:.3f} ({topk_avg_meters[j].avg:.3f})\t'

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'{prec_str}')

    topk_prec_avg = [avg_meter.avg for avg_meter in topk_avg_meters]
    prec_str = "  ".join([f'Prec {topk[i]} ({topk_prec_avg[i]:.3f})' for i in range(len(topk))])
    print(f' * {prec_str}\n')
    return topk_prec_avg


def conv_penalty(model, device, penalty, mode):
    if not ('filter' in mode or 'channel' in mode):
        return 0.0
    penalty_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
            conv_data = module.weight.data.cpu().numpy()
            if 'filter' in mode:
                left_filter_indice = model.conv2leftIndiceDict[name]
            else:
                left_filter_indice = list(range(conv_data.shape[0]))
            num_of_left_filters = len(left_filter_indice)
            penalty_filters = 0
            for i in range(num_of_left_filters - 1):
                if 'filter' in mode:
                    penalty_filters += penalty(
                        module.weight[left_filter_indice[i], :, :, :],
                        module.weight[left_filter_indice[i+1], :, :, :]
                    )
                elif 'channel' in mode:
                    left_channel_indice = model.conv2leftIndiceDict[name]
                    penalty_channels = 0
                    for j in range(len(left_channel_indice) - 1):
                        penalty_channels += penalty(
                            module.weight[i, left_channel_indice[j], :, :],
                            module.weight[i, left_channel_indice[j+1], :, :]
                        )
                    penalty_filters += penalty_channels
            penalty_filters /= (num_of_left_filters - 1)
            penalty_layers.append(penalty_filters)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.to(device)


def fc_penalty(model, device, penalty):
    penalty_fc1 = penalty_fc2 = penalty_fc3 = 0
    for i in range(int(model.partition_size['fc1']) - 1):
        penalty_fc1 += penalty(
            model.fc1.weight[
               i*model.block_row_size1: (i+1)*model.block_row_size1: 1,
               i*model.block_col_size1: (i+1)*model.block_col_size1: 1
            ],
            model.fc1.weight[
                (i+1)*model.block_row_size1: (i+2)*model.block_row_size1: 1,
                (i+1)*model.block_col_size1: (i+2)*model.block_col_size1: 1
            ])
    for i in range(int(model.partition_size['fc2']) - 1):
        penalty_fc2 += penalty(
            model.fc2.weight[
                i*model.block_row_size2: (i+1)*model.block_row_size2: 1,
                i*model.block_col_size2: (i+1)*model.block_col_size2: 1
            ],
            model.fc2.weight[
                (i+1)*model.block_row_size2: (i+2)*model.block_row_size2: 1,
                (i+1)*model.block_col_size2: (i+2)*model.block_col_size2: 1
            ])
    for i in range(int(model.partition_size['fc3']) - 1):
        penalty_fc3 += penalty(
            model.fc3.weight[
                i*model.block_row_size3: (i+1)*model.block_row_size3: 1,
                i*model.block_col_size3: (i+1)*model.block_col_size3: 1
            ],
            model.fc3.weight[
                (i+1)*model.block_row_size3: (i+2)*model.block_row_size3: 1,
                (i+1)*model.block_col_size3: (i+2)*model.block_col_size3: 1
            ])

    penalty_fc1 = penalty_fc1 / (int(model.partition_size['fc1']) - 1)
    penalty_fc2 = penalty_fc2 / (int(model.partition_size['fc2']) - 1)
    penalty_fc3 = penalty_fc3 / (int(model.partition_size['fc3']) - 1)
    penalty = 0.33 * (penalty_fc1 + penalty_fc2 + penalty_fc3)
    return penalty.to(device)


def get_layers_penalty(model, penalty, args, tok):
    all_penalty = fc_penalty_ = conv_penalty_ = 0.0
    if args.model_mode != 'c' and args.alpha != 0:
        fc_penalty_ = fc_penalty(model, args.device, penalty)
        all_penalty += args.alpha * fc_penalty_
    if args.model_mode != 'd' and tok == "prune_retrain" and args.beta != 0:
        conv_penalty_ = conv_penalty(model, args.device, penalty, args.prune_mode)
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


def train(train_loader, model, optimizer, epoch, args, tok=""):
    """ Train one epoch. """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    all_penalties = AverageMeter()
    fc_penalties = AverageMeter()
    conv_penalties = AverageMeter()
    criterion = nn.CrossEntropyLoss().to(args.device)
    penalty = nn.MSELoss(reduction='sum').to(args.device)
    # penalty = nn.L1Loss(reduction='sum').to(args.device)
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)  # measure data loading time

        target = target.to(args.device)
        input_var = torch.autograd.Variable(input).to(args.device)
        target_var = torch.autograd.Variable(target).to(args.device)

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
        if tok == "prune_retrain":
            for name, p in model.named_parameters():
                if (args.model_mode == 'd' and 'conv' in name or
                        args.model_mode == 'c' and 'fc' in name or
                        'mask' in name or
                        'bias' in name):
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
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'Prec {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Layer penalty {all_penalties.avg:.3f} ( fc: {fc_penalties.avg:.3f} , conv: {conv_penalties.avg:.3f} )')


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_drop_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
