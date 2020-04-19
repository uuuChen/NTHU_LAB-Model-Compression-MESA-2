import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import util
from prune import PruningModule, MaskedConv2d

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# ---------------------------------------- AlexNet --------------------------------------------
class AlexNet(PruningModule):
    def __init__(self, num_of_class=100, mask_flag=False):
        super(AlexNet, self).__init__()
        conv2d = MaskedConv2d if mask_flag else nn.Conv2d

        self.conv1 = conv2d(3, 64, kernel_size=(11, 11), stride=4, padding=5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = conv2d(64, 192, kernel_size=(5, 5), padding=2)
        self.conv2_bn = nn.BatchNorm2d(192)
        self.conv3 = conv2d(192, 384, kernel_size=(3, 3), padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)
        self.conv4 = conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_of_class)

        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv4(out)
        out = self.conv4_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv5(out)
        out = self.conv5_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


def inverse_permutation(p):
    s = torch.empty(p.size(), dtype=torch.long)
    index = 0
    for i in p:
        s[i] = index
        index += 1
    return s


def mask(in_weight, out_weight, partition_size):
    seed = 10
    np.random.seed(seed)

    row = out_weight
    col = in_weight

    row_temp = np.random.permutation(row)  # {1,2,5,6,....}
    col_temp = np.random.permutation(col)

    row_permu = torch.from_numpy(row_temp).long()
    col_permu = torch.from_numpy(col_temp).long()

    row = row // partition_size
    col = col // partition_size

    a = np.full((row, col), 1, dtype=int)
    binary_mask = np.kron(np.eye(partition_size), a)
    real_binary_mask = np.pad(binary_mask, ((0, out_weight % partition_size), (0, in_weight % partition_size)),
                              'constant', constant_values=(0, 0))  # to make it able to divide
    return row, col, row_permu, col_permu, torch.from_numpy(real_binary_mask)


class AlexNet_mask(PruningModule):
    def __init__(self, partitions, num_of_class=100, mask_flag=False):
        super(AlexNet_mask, self).__init__()
        conv2d = MaskedConv2d if mask_flag else nn.Conv2d
        self.partition_size = partitions

        def fc1_hook(grad):
            return grad * self.mask1.float().cuda()

        def fc2_hook(grad):
            return grad * self.mask2.float().cuda()

        def fc3_hook(grad):
            return grad * self.mask3.float().cuda()

        self.conv1 = conv2d(3, 64, kernel_size=(11, 11), stride=4, padding=5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = conv2d(64, 192, kernel_size=(5, 5), padding=2)
        self.conv2_bn = nn.BatchNorm2d(192)
        self.conv3 = conv2d(192, 384, kernel_size=(3, 3), padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)
        self.conv4 = conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)

        # self.fc1= nn.Linear(12544,4096)
        self.fc1 = nn.Linear(256, 4096)
        self.block_row_size1, self.block_col_size1, self.rowp1, self.colp1, self.mask1 = (
            mask(256, 4096, int(partitions['fc1'])))
        self.invrow1 = inverse_permutation(self.rowp1)
        self.invcol1 = inverse_permutation(self.colp1)
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight * self.mask1.float())
        self.fc1.weight.register_hook(fc1_hook)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.block_row_size2, self.block_col_size2, self.rowp2, self.colp2, self.mask2 = (
            mask(4096, 4096, int(partitions['fc2'])))
        self.invrow2 = inverse_permutation(self.rowp2)
        self.invcol2 = inverse_permutation(self.colp2)
        self.fc2.weight = torch.nn.Parameter(self.fc2.weight * self.mask2.float())
        self.fc2.weight.register_hook(fc2_hook)

        self.fc3 = nn.Linear(4096, num_of_class)
        self.block_row_size3, self.block_col_size3, self.rowp3, self.colp3, self.mask3 = (
            mask(4096, num_of_class, int(partitions['fc3'])))
        self.invrow3 = inverse_permutation(self.rowp3)
        self.invcol3 = inverse_permutation(self.colp3)
        self.fc3.weight = torch.nn.Parameter(self.fc3.weight * self.mask3.float())
        self.fc3.weight.register_hook(fc3_hook)

        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = F.relu(out)        
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv4(out)
        out = self.conv4_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv5(out)
        out = self.conv5_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


# ---------------------------------------- VGGNet --------------------------------------------
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(PruningModule):
    def __init__(self, features, num_classes=100, init_weights=True):
        super(VGG, self).__init__()
        self.conv_layers = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # fixed fully connected layer input
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


def get_model(args):
    if args.model_mode == 'c':
        if args.use_model == 'alex':
            args.prune_rates = [0.16, 0.62, 0.65, 0.63, 0.63]
            model = AlexNet(100, mask_flag=True).to(args.device)
        elif args.use_model == 'vgg16':
            # args.prune_rates = [0.42, 0.78, 0.66, 0.64, 0.47, 0.76, 0.58, 0.68, 0.73, 0.66, 0.65, 0.71, 0.64]  # deepC
            args.prune_rates = [0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.75, 0.75, 0.75, 0.75, 0.75]  # PFEC
            model = vgg16().to(args.device)
        else:
            raise Exception
    else:
        model = AlexNet_mask(args.partition, 100, mask_flag=True).to(args.device)
    if os.path.isfile(f"{args.load_model}"):
        model, args.best_prec1 = util.load_checkpoint(model, f"{args.load_model}", args)
        print("-------load " + f"{args.load_model} ({args.best_prec1:.3f})----")
    return model



