import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class PruningModule(Module):
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

    def prune_step(self, prune_rates, mode='filter'):
        dim = 0
        conv_idx = 0
        prune_indices = None
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
                if mode == 'filter':
                    if dim == 1:
                        self._prune_by_indices(module, dim, prune_indices)
                        dim ^= 1
                    prune_indices = self._get_prune_indices(module.weight.data, prune_rates[conv_idx], mode=mode)
                    self._prune_by_indices(module, dim, prune_indices)
                    dim ^= 1
                elif mode == 'channel':
                    dim = 1
                    prune_indices = self._get_prune_indices(module.weight.data, prune_rates[conv_idx], mode=mode)
                    self._prune_by_indices(module, dim, prune_indices)
                conv_idx += 1

            elif isinstance(module, torch.nn.BatchNorm2d):
                if mode == 'filter' and dim == 1:
                    self._prune_by_indices(module, 0, prune_indices)

    def _get_prune_indices(self, conv_tensor, prune_rate, mode='filter'):
        if conv_tensor.is_cuda:
            conv_tensor = conv_tensor.cpu()
        sum_of_objects = None
        object_nums = None
        if mode == 'filter':
            sum_of_objects = torch.sum(torch.abs(conv_tensor.reshape(conv_tensor.size(0), -1)), 1)
            object_nums = conv_tensor.size(0)
        elif mode == 'channel':
            perm_conv_tensor = conv_tensor.permute(1, 0, 2, 3)  # (fn, cn, kh, kw) => (cn, fn, kh, kw)
            sum_of_objects = torch.sum(torch.abs(perm_conv_tensor.reshape(perm_conv_tensor.size(0), -1)), 1)
            object_nums = conv_tensor.size(1)
        _, object_indices = torch.sort(sum_of_objects)
        pruned_object_nums = round(object_nums * prune_rate)
        return object_indices[:pruned_object_nums].tolist()

    def _prune_by_indices(self, module, dim, indices):
        if dim == 0:
            if len(module.weight.size()) == 4:  # conv layer
                module.weight.data[indices, :, :, :] = 0.0
            elif len(module.weight.size()) == 1:  # conv_bn layer
                module.weight.data[indices] = 0.0
            module.bias.data[indices] = 0.0
        elif dim == 1:
            module.weight.data[:, indices, :, :] = 0.0  # only happened to conv layer


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
        # self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
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




