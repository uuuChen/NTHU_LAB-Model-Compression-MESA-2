import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import warnings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from deep_compression.huffman_encode import deep_huffman_encode
from mpd.huffman_encode import mpd_huffman_encode

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
#from torchsummary import summary

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


parser = argparse.ArgumentParser(description='PyTorch cifar100 Training')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--epochs', '-ep', default=100, type=int, metavar='N',
                    help='number of total initial epochs to run (default: 100)')
parser.add_argument('--batch-size', '-bs', default=256, type=int,metavar='N',
                    help='mini-batch size (default: 256)')

# ---------- log file --------------------------------------------------------------------------------
parser.add_argument('--log', type=str, default='huffman_log.txt',
                    help='log file name')
# ---------- log file --------------------------------------------------------------------------------
parser.add_argument('--log-detail', type=str, default='huffman_detail_log.txt',
                    help='log file name')
# ---------- partition size for fc1, fc2, fc3 --------------------------------------
parser.add_argument("--partition",'-p', dest="partition",
                    action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...")
# ---------- quantize bits ---------------------------------------------------------
parser.add_argument('--bits', '-b',
                    action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='partition size of fc layer (eg. conv=8,fc=5)')
# ---------- index bits ---------------------------------------------------------
parser.add_argument('--index-bits', '-ib', default=5, type=int,
                    help='quantizatize bit')
# ------------- alpha ---------------------------------------------------------------
parser.add_argument('--alpha','-al', default=0.1, type=float, metavar='M',
                    help='alpha(default=0.1)')
# ---------- base dir --------------------------------------------------------------
parser.add_argument('--save-dir','-sd', type=str,default='model_default',
                    help='model store path(defalut="model_default")')
# ---------- load file -------------------------------------------------------------
parser.add_argument('--load-model-init','-lmi', type=str,default='checkpoint_default.tar',
                    help='load exist checkpoint file')
# ---------- load file -------------------------------------------------------------
parser.add_argument('--load-model-deepcompress','-lmd', type=str,default='checkpoint_default.tar',
                    help='load exist checkpoint file')
# ---------- load file -------------------------------------------------------------
parser.add_argument('--load-model-mpdonlyp','-lmp', type=str,default='checkpoint_default.tar',
                    help='load exist checkpoint file')
# ---------- load file -------------------------------------------------------------
parser.add_argument('--load-model-mpd','-lmm', type=str,default='checkpoint_default.tar',
                    help='load exist checkpoint file')
# ---------- load file -------------------------------------------------------------
parser.add_argument('--load-model-onlyq','-lmq', type=str,default='checkpoint_default.tar',
                    help='load exist checkpoint file')
# ---------- prune mode -----------------------------------------------------
parser.add_argument('--prune-mode', '-pm', default='filter', type=str, metavar='M',
                    help='filter: pruned by filter percentile: pruned by percentile channel: pruned by channel\n')

# ---------- load model ---------------------------------------------------------
parser.add_argument('--train-mode', '-tm', type=str, default='',
                    help='saved quantized model')
# ---------- dnn or cnn or all -----------------------------------------------------
parser.add_argument('--model-mode', '-mm', default='d', type=str, metavar='M',
                    help='d:only qauntize dnn c:only qauntize cnn a:all qauntize\n')


def main():

    os.makedirs(f'{args.save_dir}', exist_ok=True)
#--------------------different method----------------------------------------

    # print('FC')
    # acc_init, fc_org_total_init,  fc_compressed_total_init = deep_huffman_encode(val_loader ,args.load_model_init ,"t" ,False ,args)
    # print('DeepC')
    # acc_deep, fc_org_total_deep,  fc_compressed_total_deep = deep_huffman_encode(val_loader ,args.load_model_deepcompress ,"t" ,True , args)
    # print('FC+Q+H')
    # acc_onlyq, fc_org_total_onlyq,  fc_compressed_total_onlyq = deep_huffman_encode(val_loader ,args.load_model_onlyq ,"f" ,True ,args)
    # print('MPDC')
    # acc_mpdp, fc_org_total_mpdp,  fc_compressed_total_mpdp, fc_compressed_without_edit_mpdp, edit_distance_list, fc_t, fc_d, layer_compressed_dic, layer_org_dic_mpdp =  mpd_huffman_encode(val_loader ,args.load_model_mpdonlyp ,args)

    print('MPDC+penalty')
    alpha_list = []
    compressed_alpha = []
    compressed_without_edit_alpha = []
    acc_alpha = []
    edit_distance_alpha = []
    # alpha = float(args.load_model_mpd.split("_")[4])
    # split_list = args.load_model_mpd.split("_")
    # alpha_list.append(alpha)

    acc_deep, org_total_deep, compressed_total_deep = deep_huffman_encode(val_loader, "checkpoint_quantized_percentile_re_alpha_0.1_19.tar", "t", True, args)
    print(int(round((org_total_deep / compressed_total_deep))))
    acc_mpd, org_total_mpd,  compressed_total_mpd, fc_compressed_without_edit_mpd, edit_distance_list, fc_t, fc_d, layer_compressed_dic, layer_org_dic = mpd_huffman_encode(val_loader, args.load_model_mpd, args)
    print(int(round((org_total_mpd / compressed_total_mpd))))

    # compressed_alpha.append(fc_compressed_total_mpd)
    # compressed_without_edit_alpha.append(fc_compressed_without_edit_mpd)
    # acc_alpha.append(acc_mpd.item())
    #
    # best_alpha_acc = 0
    # best_alpha = -1
    # alpha=0.1
    # split_list[4] = str(alpha)
    # best_mpd_layer_compression_dic = layer_compressed_dic
    #
    # while alpha <=0.1:
    #     alpha_list.append(alpha)
    #     acc_mpd, fc_org_total_mpd,  fc_compressed_total_mpd , fc_compressed_without_edit_mpd, edit_distance_list, fc_t, fc_d, layer_compressed_dic, layer_org_dic= mpd_huffman_encode(val_loader ,"_".join(split_list) ,args)
    #     compressed_alpha.append(fc_compressed_total_mpd)
    #     compressed_without_edit_alpha.append(fc_compressed_without_edit_mpd)
    #     edit_distance_alpha.append(edit_distance_list)
    #
    #     acc_alpha.append(acc_mpd.item())
    #     alpha*=10
    #     split_list[4] = str(alpha)
    #     if acc_mpd > best_alpha_acc:
    #         best_alpha_acc = acc_mpd
    #         best_alpha = alpha
    #         fc_compressed_total_mpd_alpha = fc_compressed_total_mpd
    #     best_alpha_acc = acc_mpd
    #     best_alpha = alpha
    #     fc_compressed_total_mpd_alpha = fc_compressed_total_mpd
    #     best_mpd_layer_compression_dic = layer_compressed_dic

    # alpha_objects = tuple(alpha_list)
    # y_pos = np.arange(len(alpha_objects))
    # for k, v in layer_compressed_dic.items():
    #     print(f'{k}\t{layer_org_dic[k]}\t{v}')

# #----------- plot fc space on different alpha with editgraph------------------
#     compressed_rate=[]
#     compressed_rate[:] = [int(round(fc_org_total_deep/x)) for  x in compressed_alpha]
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.bar(y_pos, compressed_rate, align='center',color='Blue', label='Compression Rate')
#     for a,b in zip(y_pos, compressed_rate):
#         ax1.text(a-0.25, b+2, str(b)+'x',fontsize=10)
#     plt.xticks(y_pos, alpha_objects)
#     ax1.set_ylim(0, 250)
#     ax1.set_xlabel('Alpha', fontsize=14)
#     ax1.set_ylabel('Compression Rate', fontsize=14)
#     ax2 = ax1.twinx()
#     ax2.plot(y_pos, acc_alpha, '-ro', label='Accuracy')
#     for a,b in zip(y_pos,acc_alpha):
#         ax2.text(a-0.25, b+2, "{0:.2f}".format(b),fontsize=10)
#     ax2.set_ylim(0, 100)
#     ax2.set_ylabel('Accuracy (%)', fontsize=14)
#     ax2.set_xlabel('Alpha', fontsize=14)
#     ax1.plot(np.nan, '-ro', label = 'Accuracy')
#     ax1.legend(loc='lower right')
#     plt.savefig(f'{args.save_dir}/Alexnet_fc_space_penalty.png')
#     plt.close()
#
# #----------- plot comparison of different compression methods ------------------
#
#     compressed_list = [int(round(fc_org_total_deep / fc_org_total_deep)),
#                        int(round(fc_org_total_deep / fc_compressed_total_onlyq)),
#                        int(round(fc_org_total_deep / fc_compressed_total_deep)),
#                        int(round(fc_org_total_deep / sum(layer_org_dic_mpdp.values()))),
#                        int(round(fc_org_total_deep / compressed_without_edit_alpha[0])),
#                        int(round(fc_org_total_deep / fc_compressed_total_mpd_alpha))]
#
#     acc_list = [acc_init, acc_onlyq, acc_deep, acc_mpdp, acc_alpha[0], best_alpha_acc]
#
#     method_objects = ('FC','FC+Q+H','DeepC','MPDC','MPDC+Q+H', 'MESA')
#     method_pos = np.arange(len(method_objects))
#     plt.bar(method_pos, compressed_list, align='center',color='Blue')
#     for a,b in zip(method_pos, compressed_list):
#         plt.text(a-0.25, b+0.1, str(b))
#     plt.xticks(method_pos, method_objects)
#     plt.yscale('log')
#     plt.xlabel('compress algorithm')
#     plt.ylabel('compression rate')
#     plt.title('Compression results on AlexNet')
#     plt.savefig(f'{args.save_dir}/compress_result.png')
#     plt.close()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    global args, best_prec1, train_loade, val_loader
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    if use_cuda:
        print("Using CUDA!")
        torch.cuda.manual_seed(args.seed)
    else:
        print('Not using CUDA!!!')

    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # load
    # kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    # data = '../data/ILSVRC2012_train_and_val/'
    # traindir = os.path.join(data, 'train')
    # valdir = os.path.join(data, 'valid')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(227),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # train_sampler = None
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=8, pin_memory=True, sampler=train_sampler)
    #
    # print(train_dataset[0][0].size())
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(227),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=8, pin_memory=True)

    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    main()
