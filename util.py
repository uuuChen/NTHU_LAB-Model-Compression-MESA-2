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
        print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
    else:
        print ("=> no checkpoint found at '{}'".format(file))
    return model


def save_masked_checkpoint(model, mode, is_best, best_prec1, epoch, args):
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}_alpha_{}_{}.tar'.format(mode,args.alpha,epoch)))

    return model


def layer2torch(folder_path, model):
    for name, modules in model.named_children():
        torch.save(modules.weight.data, f"{folder_path}/{name}.txt")


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
                weights = np.append([weights], [mat], axis=0)
            else:
                weights = np.append(weights, [mat], axis=0)
        weights = weights.reshape(shape)
        weight_list.append(weights)
    return weight_list


def save_parameters(file_folder, quantized_parameters):
    for i in range(len(quantized_parameters)):
        layer_weight = quantized_parameters[i]
        shape = layer_weight.shape
        if len(shape) > 1:
            f = open(file_folder + '/layer_' + str(i) + '.txt', 'a')
            if len(shape) == 2:
                for weight in layer_weight:
                    for j in range(len(weight)):
                        f.write(str(weight[j]) + "\t")
                    f.write("\n")
            else:
                f.write(str(shape) + "\n")
                for filter_weight in layer_weight:
                    for weight in filter_weight:
                        weight = weight.reshape(-1)
                        for a_weight in weight:
                            f.write(str(a_weight) + "\t")
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
        log(log_file, f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%)'
                      f' | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | '
              f'total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    log(log_file, f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : '
                  f'{total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x '
          f' ({100 * (total-nonzero) / total:6.2f}% pruned)')


def initial_train(model, args, train_loader, val_loader, tok, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    best_prec1 = 0
    best_prec5 = 0
    cudnn.benchmark = True
    if tok == 'prune_retrain':
        epochs = args.reepochs
        args.lr = 0.0001
    else:
        epochs = args.epochs
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    initial_optimizer_state_dict = optimizer.state_dict()
    optimizer.load_state_dict(initial_optimizer_state_dict)  # Reset the optimizer

    print('start epoch', args.start_epoch)
    print('epoch', epochs)
    for epoch in range(args.start_epoch, epochs):
        optimizer = adjust_learning_rate(optimizer, epoch, args)
        print('in_epoch', epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args.alpha, args, tok)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, args, topk=(1, 5))

        # remember best prec1 and save checkpoint
        if best_prec1 < prec1 or tok == "prune_retrain":
            model = save_masked_checkpoint(model, tok, True, best_prec1, epoch, args)
            log(f"{args.save_dir}/{args.log}", f"[epoch {epoch}]")
            log(f"{args.save_dir}/{args.log}", f"initial_accuracy\t{prec1}")
            log(f"{args.save_dir}/{args.log}", f"initial_top5_accuracy\t{prec5}")
            best_prec1 = prec1

        if args.prune_mode == 'filter-gm' and tok == "initial":
            if epoch % args.prune_interval == 0 or epoch == epoch-1:  # 1 is hyper parameter
                prune_rates = model.get_conv_actual_prune_rates(args.prune_rates)
                model.prune_step(prune_rates, mode=args.prune_mode)

    model = save_masked_checkpoint(model, tok, True, best_prec5, epochs, args)
    return model


def quantized_retrain(model, args, quantized_index_list, quantized_center_list, train_loader, val_loader, use_cuda=True):
    criterion = nn.CrossEntropyLoss().cuda()
    best_prec5 = 0
    args.lr = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.qauntize_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        model.train()  # switch to train mode
        end = time.time()
        optimizer = adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            target = target.cuda()
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()
            k = 0

            for name, p in model.named_parameters():
                if args.model_mode == 'd' and 'conv' in name:
                    continue
                if args.model_mode == 'c' and 'fc' in name:
                    continue
                if 'mask' in name or 'bias' in name or 'bn' in name:
                    continue

                quantized_index = quantized_index_list[k]
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_center_array = []
                for j in range(2**args.bits):
                    grad_by_index = grad_tensor[quantized_index == j]
                    grad_center = np.mean(grad_by_index)
                    grad_center_array.append(grad_center)

                grad_center_array = np.array(grad_center_array)
                grad_tensor = grad_center_array[quantized_index]
                grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).cuda()
                k += 1

            optimizer.step()
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1))

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, args, topk=(1, 5))
        # prec5 = validate(val_loader, model, args, topk=(5,))
        model = save_masked_checkpoint(model, 'quantized_re', True, prec5, epoch, args)
        log(f"{args.save_dir}/{args.log}", f"[epoch {epoch}]")
        log(f"{args.save_dir}/{args.log}", f"initial_accuracy\t{prec1}")
        log(f"{args.save_dir}/{args.log}", f"initial_top5_accuracy\t{prec5}")

    model = save_masked_checkpoint(model,'quantized_re', True, best_prec5, args.reepochs, args)
    return model


def test(model, test_loader, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({accuracy:.2f}%)')
    return accuracy


def validate(val_loader, model, args, topk=(1,), tok=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    topk_list = [AverageMeter() for _ in range(len(topk))]
    criterion = nn.CrossEntropyLoss().cuda()
    penalty = nn.MSELoss(reduction='sum').cuda()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if tok == "prune_retrain":
            layer_penalty = get_layer_penalty(model, penalty, args)
            loss += layer_penalty
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        preck = accuracy(output.data, target, topk=topk)
        losses.update(loss.item(), input.size(0))
        prec_str = str()
        for j in range(len(topk)):
            topk_list[j].update(preck[j], input.size(0))
            prec_str += f'Prec {topk[j]} [{topk_list[j].val:.3f}) ({topk_list[j].avg:.3f})]\t'

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print(f'\nTest: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'{prec_str}')

    topk_prec_avg = [topk.avg for topk in topk_list]
    prec_str = "  ".join([f'Prec {topk[i]} ({topk_prec_avg[i]:.3f})' for i in range(len(topk))])
    print(f' * {prec_str}\n')
    return topk_prec_avg


def conv_penalty(model, penalty, mode):
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
                    cur_filter_idx = left_filter_indice[i]
                    next_filter_idx = left_filter_indice[i+1]
                    penalty_filters += penalty(
                        module.weight[cur_filter_idx, :, :, :],
                        module.weight[next_filter_idx, :, :, :]
                    )
                elif 'channel' in mode:
                    left_channel_indice = model.conv2leftIndiceDict[name]
                    penalty_channels = 0
                    for j in range(len(left_channel_indice) - 1):
                        cur_channel_idx = left_channel_indice[j]
                        next_channel_idx = left_channel_indice[j+1]
                        penalty_channels += penalty(
                            module.weight[i, cur_channel_idx, :, :],
                            module.weight[i, next_channel_idx, :, :]
                        )
                    penalty_filters += penalty_channels
            penalty_filters /= (num_of_left_filters - 1)
            penalty_layers.append(penalty_filters)
    penalty = torch.mean(torch.stack(penalty_layers))
    return penalty.cuda()


def fc_penalty(model, penalty):
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
    return penalty.cuda()


def get_layer_penalty(model, penalty, args):
    layer_penalty = 0.0
    if args.model_mode != 'c':
        layer_penalty += args.alpha * fc_penalty(model, penalty)
    if args.model_mode != 'd':
        layer_penalty += args.beta * conv_penalty(model, penalty, args.prune_mode)
    return layer_penalty


def accuracy(output, target, topk=(1,)):
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


def train(train_loader, model, optimizer, epoch, alpha, args, tok=""):
    """ Run one train epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()
    penalty = nn.MSELoss(reduction='sum').cuda()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output and loss
        output = model(input_var)
        loss = criterion(output, target_var)
        if tok == "prune_retrain":
            layer_penalty = get_layer_penalty(model, penalty, args)
            loss += layer_penalty

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()

        if tok == "prune_retrain":
            for name, p in model.named_parameters():
                if args.model_mode == 'd' and 'conv' in name:
                    continue
                if args.model_mode == 'c' and 'fc' in name:
                    continue
                if 'mask' in name:
                    continue
                if 'bias' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).cuda()

        optimizer.step()
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            # print(layer_penalty)


def get_method_str(args):
    method_list = list()
    if args.model_mode != 'c':
        method_list.append(f'alpha_{args.alpha}')
    if args.model_mode != 'd':
        method_list.append(f'beta_{args.beta}')
    method_str = '_'.join(method_list)
    return method_str


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
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
