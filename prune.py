import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class PruningModule(Module):
    def __init__(self):
        super(PruningModule, self).__init__()
        self.conv2pruneIndiceDict = dict()
        self.conv2leftIndiceDict = dict()

    def prune_by_percentile(self, layerlist, q=5.0):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        for name, module in self.named_modules():
            if name in layerlist:
                tensor = module.weight.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), q)
                print(f'Pruning {name} with threshold : {percentile_value}')
                self._prune_by_threshold(module, percentile_value)

    def prune_by_std(self, s, layerlist):
        """
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        for name, module in self.named_modules():
            if name in layerlist:
                # ['conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8','conv9','conv10','conv11','conv12','
                # conv13','fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                self._prune_by_threshold(module, threshold)

    def _prune_by_threshold(self, module, threshold):
        weight_dev = module.weight.device
        mask_dev = module.mask.device

        # Convert Tensors to numpy and calculate
        tensor = module.weight.data.cpu().numpy()
        mask = module.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)

        # Apply new weight and mask
        module.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        module.mask.data = torch.from_numpy(new_mask).to(mask_dev)

    def prune_step(self, prune_rates, mode='filter-norm'):
        dim = 0
        conv_idx = 0
        prune_indices = None
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
                if 'filter' in mode:
                    if dim == 1:
                        self._prune_by_indice(module, dim, prune_indices)
                        dim ^= 1
                    prune_indices = self._get_prune_indice(module.weight.data, prune_rates[conv_idx], mode=mode)
                    self._prune_by_indice(module, dim, prune_indices)
                    dim ^= 1
                elif 'channel' in mode:
                    dim = 1
                    prune_indices = self._get_prune_indice(module.weight.data, prune_rates[conv_idx], mode=mode)
                    self._prune_by_indice(module, dim, prune_indices)
                conv_idx += 1
            elif isinstance(module, torch.nn.BatchNorm2d):
                if 'filter' in mode and dim == 1:
                    self._prune_by_indice(module, 0, prune_indices)

    def _get_prune_indice(self, conv_tensor, prune_rate, mode='filter-norm'):
        if conv_tensor.is_cuda:
            conv_tensor = conv_tensor.cpu()
        conv_arr = conv_tensor.numpy()
        sum_of_objects = None
        object_nums = None
        if mode == 'filter-norm':
            sum_of_objects = np.sum(np.abs(conv_arr.reshape(conv_arr.shape[0], -1)), 1)
            object_nums = conv_arr.shape[0]
        elif mode == 'channel-norm':
            perm_conv_arr = np.transpose(conv_arr, (1, 0, 2, 3))  # (fn, cn, kh, kw) => (cn, fn, kh, kw)
            sum_of_objects = np.sum(np.abs(perm_conv_arr.reshape(perm_conv_arr.shape[0], -1)), 1)
            object_nums = conv_tensor.shape[1]
        elif mode == 'filter-gm':
            filters_flat_arr = conv_arr.reshape(conv_arr.shape[0], -1)
            sum_of_objects = np.array([np.sum(np.power(filters_flat_arr-arr, 2)) for arr in filters_flat_arr])
            object_nums = conv_arr.shape[0]
        object_indice = np.argsort(sum_of_objects)
        pruned_object_nums = round(object_nums * prune_rate)
        pruned_indice = np.sort(object_indice[:pruned_object_nums])
        # print(list(zip(pruned_indice, sum_of_objects[object_indice])))
        return pruned_indice

    def _prune_by_indice(self, module, dim, indices):
        if dim == 0:
            if len(module.weight.size()) == 4:  # conv layer etc.
                module.weight.data[indices, :, :, :] = 0.0
            elif len(module.weight.size()) == 1:  # conv_bn layer etc.
                module.weight.data[indices] = 0.0
            module.bias.data[indices] = 0.0
        elif dim == 1:
            module.weight.data[:, indices, :, :] = 0.0  # only happened to conv layer, so its dimension is 4

    def get_conv_actual_prune_rates(self, prune_rates, mode='filter', print_log=False):
        """ Suppose the model prunes some filters (filters, :, :, :) or channels (:, channels, :, :). """
        conv_idx = 0
        prune_filter_nums = None
        new_pruning_rates = list()
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
                conv_shape = module.weight.shape
                filter_nums = conv_shape[0]
                channel_nums = conv_shape[1]

                target_filter_prune_rate = target_channel_prune_rate = prune_rates[conv_idx]

                # If the filter of the previous layer is prune, the channel corresponding to this layer must also be
                # prune
                # Conv Shape: (fn, cn, kh, kw)
                # (1 - prune_rate) = ((fn * (1 - new_prune_rate)) * (cn - Prune_filter_nums) * kh * kw) / (fn * cn * kh
                # * kw)
                if conv_idx != 0:
                    target_filter_prune_rate = (
                            1 - ((1 - target_filter_prune_rate) * (channel_nums / (channel_nums - prune_filter_nums)))
                    )
                prune_filter_nums = round(filter_nums * target_filter_prune_rate)
                actual_filter_prune_rate = prune_filter_nums / filter_nums
                filter_bias = abs(actual_filter_prune_rate - target_filter_prune_rate)

                prune_channel_nums = round(channel_nums * target_channel_prune_rate)
                actual_channel_prune_rate = prune_channel_nums / channel_nums
                channel_bias = abs(actual_channel_prune_rate - target_channel_prune_rate)

                if mode == 'filter':
                    if print_log:
                        print(f'{name:6} | original filter nums: {filter_nums:4} | prune filter nums: '
                              f'{prune_filter_nums:4} | target filter prune rate: {target_filter_prune_rate * 100.:.2f}'
                              f'% | actual filter prune rate : {actual_filter_prune_rate * 100.:.2f}% | filter bias: '
                              f'{filter_bias * 100.:.2f}%\n')
                    new_pruning_rates.append(actual_filter_prune_rate)

                elif mode == 'channel':
                    if print_log:
                        print(f'{name:6} | original channel nums: {channel_nums:4} | prune channel nums: '
                              f'{prune_channel_nums:4} | target channel prune rate: '
                              f'{target_channel_prune_rate * 100.:.2f}% | actual channel prune rate: '
                              f'{actual_channel_prune_rate * 100.:.2f}% | channel bias: {channel_bias * 100.:.2f}%\n')

                    new_pruning_rates.append(actual_channel_prune_rate)

                conv_idx += 1

        return new_pruning_rates

    def set_conv_prune_indice_dict(self, mode='filter-norm'):
        pruned_indice = None
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
                conv_arr = module.weight.data.cpu().numpy()
                if 'filter' in mode:
                    pruned_indice = np.where(np.sum(conv_arr.reshape(conv_arr.shape[0], -1), axis=1) == 0)[0]
                elif 'channel' in mode:
                    perm_conv_arr = np.transpose(conv_arr, (1, 0, 2, 3))  # (fn, cn, kh, kw) => (cn, fn, kh, kw)
                    pruned_indice = np.where(np.sum(perm_conv_arr.reshape(perm_conv_arr.shape[0], -1), axis=1) == 0)[0]
                left_indice = list(set(range(conv_arr.shape[0])).difference(pruned_indice))
                self.conv2pruneIndiceDict[name] = pruned_indice
                self.conv2leftIndiceDict[name] = left_indice


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.mask = Parameter(torch.ones(in_channels, out_channels // groups, *kernel_size), requires_grad=False)
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.mask = Parameter(torch.ones(out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class MaskedConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MaskedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight*self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)




