# System
import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import warnings
from PIL import ImageFile

# Custom
import models
import util
import data_loader
from quantization import apply_weight_sharing
from mesa2_encoder import mesa2_huffman_encode_model
from deepc_encoder import deepc_huffman_encode_model
from prune import MaskedConv2d


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


parser = argparse.ArgumentParser(description='MESA2 Training')

##############################
# MESA setting
##############################
parser.add_argument("--partition", '-p', dest="partition", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help='partition size of fc layer (eg. fc1=8,fc2=8,fc3=10)')


##############################
# MESA-2 setting
##############################
# ------------- general setting ----------------------------------------------------
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--start-epoch', '-sep', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-bs', default=256, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr-drop-interval', '-lr-drop', default=50, type=int,  metavar='LRD', help='learning rate drop interval (default: 50)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-pf', default=100, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--log_file_name', type=str, default='log.txt', help='log file name')
parser.add_argument('--load-model', '-lm', type=str, default='checkpoint_default.tar',  help='load exist checkpoint file')
parser.add_argument('--initial-process', '-initp', dest='initial_process', action='store_true', help='run initial process or not')
parser.add_argument('--pruning-process', '-prunep', dest='pruning_process', action='store_true', help='run pruning process or not')
parser.add_argument('--qunatize-process', '-quanp', dest='quantize_process', action='store_true', help='run quantize process or not')
parser.add_argument('--encoding-process', '-encp', dest='encoding_process', action='store_true', help='run huffman encoding process or not')
parser.add_argument('--use-model', '-um', default='alex', type=str, metavar='M', help='alex: AlexNet vgg: VggNet res:ResNet\n')
parser.add_argument('--model-mode', '-mm', default='c', type=str, metavar='M', help='d:only qauntize dnn c:only qauntize cnn a:all qauntize\n')
parser.add_argument('--use-mesa-fc-mask', '-umfm', dest='use_mesa_fc_mask', action='store_true', help='use fully connected layer mask or not')
parser.add_argument('--save-dir', '-sd', type=str, default='model_default', help='model store path(defalut="model_default")')

# ------------- initial process ----------------------------------------------------
parser.add_argument('--epochs', '-ep', default=200, type=int, metavar='N', help='number of total initial epochs to run (default: 200)')
parser.add_argument('--train-lr', '-tlr', default=0.1, type=float, metavar='TLR', help='train learning rate')

# ------------- pruning process ----------------------------------------------------
parser.add_argument('--prune-mode', '-pm', default='filter-norm', type=str, metavar='M', help='filter: pruned by filter percentile: pruned by percentile channel: pruned by channel\n')
parser.add_argument('--prune-rates', "-pr", nargs='+', type=float, default=[0.16, 0.62, 0.65, 0.63, 0.63], help='pruning rate for AlexNet conv layer (default=[0.16, 0.62, 0.65, 0.63, 0.63])')
parser.add_argument('--prune-interval', '-pi', default=1, type=int, metavar='N', help='prune interval when using prune-mode "filter-gm" (default: 1)')
parser.add_argument('--prune-retrain-epochs', '-prep', default=40, type=int, metavar='N',  help='number of pruning retrain epochs to run (default: 40)')
parser.add_argument('--prune-retrain-lr', '-prlr', default=0.0001, type=float, metavar='PRLR', help='pruning retrain learning rate')
parser.add_argument('--alpha', '-al', default=0.1, type=float, metavar='M', help='alpha (default=0.1), correspond to fc')
parser.add_argument('--beta', '-be', default=0.1, type=float, metavar='M',  help='beta (default=0.1), correspond to conv')
parser.add_argument('--conv-loss-func', '-clf', type=str, default='filter3d-delta', help='can be filter3d-delta or position-mean or matrix2d-mean (default filter3d-delta)')

# ------------- quantize process ----------------------------------------------------
parser.add_argument('--qauntize_epochs', '-qep', default=20, type=int, metavar='N',  help='number of quantize retrain epochs to run (default: 20)')
parser.add_argument('--quantize-retrain-lr', '-qrlr', default=0.0001, type=float,  metavar='QRLR', help='quantize retrain learning rate')
parser.add_argument('--bits', '-b', action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help='partition size of fc layer (eg. conv=8,fc=5)')

# ------------- encoding process  ----------------------------------------------------
parser.add_argument('--conv-delta-encoding', '-cde', dest='conv_delta_encoding', action='store_true',  help='use delta encoding on conv or not')

args = train_loader = val_loader = model = None


def check_folders_exist():
    os.makedirs(f'{args.save_dir}', exist_ok=True)


def environ_setting():
    global args, train_loader, val_loader, model
    # arguments setting
    args = parser.parse_args()
    args.method_str = util.get_method_str(args)
    args.log_file_path = os.path.join(args.save_dir, args.log_file_name)
    print(f'Method | {args.method_str}')
    args.best_prec1 = 0.0
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()

    # others setting
    if args.use_cuda:
        print("Using CUDA!")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        args.device = torch.device("cuda")
    else:
        print('Not using CUDA!')
        args.device = torch.device("cpu")
    torch.manual_seed(args.seed)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    check_folders_exist()  # make sure output folders exist
    train_loader, val_loader = data_loader.get_cifar100_data_loader(args)  # get data loader
    model = models.get_model(args)  # get model
    print(model)
    util.print_model_parameters(model)


def run_processes():
    util.log(args.log_file_path, "--------------------------configure----------------------")
    util.log(args.log_file_path, f"{args}\n")
    if args.initial_process:
        initial_process()
    if args.pruning_process:
        pruning_process()
    if args.quantize_process:
        quantize_process()
    if args.encoding_process:
        encoding_process()


def initial_process():
    util.topic_log("Initial training")
    util.print_nonzeros(model, args.log_file_path)
    util.initial_train(model, args, train_loader, val_loader, 'initial_train')
    top1_acc, top5_acc = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(args.log_file_path, f"initial_accuracy\t{top1_acc} ({top5_acc})")


def pruning_process():
    util.topic_log("Before pruning")
    util.print_nonzeros(model, args.log_file_path)
    top1_acc, top5_acc = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(args.log_file_path, f"before pruning accuracy\t{top1_acc} ({top5_acc})")

    util.topic_log("Pruning CNN")
    if args.prune_mode == 'percentile':
        conv_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, MaskedConv2d):
                model.prune_by_percentile([name], q=100*args.prune_rates[conv_idx])
                conv_idx += 1
    elif 'filter' in args.prune_mode or "channel" in args.prune_mode:
        if args.prune_mode != 'filter-gm':  # filter-gm prunes model during initial_train
            prune_rates = args.prune_rates
            if 'filter' in args.prune_mode:
                prune_rates = model.get_conv_actual_prune_rates(args.prune_rates, print_log=True)
            model.prune_step(prune_rates, mode=args.prune_mode)
        model.set_conv_prune_indices_dict()

    util.topic_log("After prune CNN")
    util.print_nonzeros(model, args.log_file_path)
    top1_acc, top5_acc = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(args.log_file_path, f"after pruning accuracy\t{top1_acc} ({top5_acc})")

    util.topic_log("Start retrain after prune CNN")
    util.initial_train(model, args, train_loader, val_loader, 'prune_retrain')

    util.topic_log("After retraining")
    util.print_nonzeros(model, args.log_file_path)
    top1_acc, top5_acc = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(args.log_file_path, f"after pruning and retrain accuracy\t{top1_acc} ({top5_acc})")


def quantize_process():
    util.topic_log("Accuracy before weight sharing")
    util.print_nonzeros(model, args.log_file_path)
    top1_acc, top5_acc = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(args.log_file_path, f"accuracy before weight sharing\t{top1_acc} ({top5_acc})")

    util.topic_log("Accuracy after weight sharing")
    quan_name2labels = apply_weight_sharing(model, args)
    top1_acc, top5_acc = util.validate(val_loader, model, args, topk=(1, 5))
    util.save_masked_checkpoint(model, "quantized", top1_acc, "initial", args)
    util.log(args.log_file_path, f"accuracy after weight sharing {args.bits}bits\t{top1_acc} ({top5_acc})")

    util.topic_log("Quantize retraining")
    util.quantized_retrain(model, args, quan_name2labels, train_loader, val_loader)
    top1_acc, top5_acc = util.validate(val_loader, model, args, topk=(1, 5))
    util.save_masked_checkpoint(model, "quantized", top1_acc, "end", args)
    util.log(args.log_file_path, f"accuracy after qauntize and retrain\t{top1_acc} ({top5_acc})")


def encoding_process():
    util.topic_log("Accuracy before huffman encoding")
    util.print_nonzeros(model, args.log_file_path)
    top1_acc, top5_acc = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(args.log_file_path, f"accuracy before huffman encoding\t{top1_acc} ({top5_acc})")

    util.topic_log("encoding")
    mesa2_huffman_encode_model(model, args)
    # deepc_huffman_encode_model(model, args)


def main():
    environ_setting()
    run_processes()


if __name__ == '__main__':
    main()
