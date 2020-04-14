import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

import AlexNet_mask
import util
import dataSet
import warnings
from net.quantization import apply_weight_sharing
from PIL import ImageFile
from prune import MaskedConv2d

ImageFile.LOAD_TRUNCATED_IMAGES = True


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


parser = argparse.ArgumentParser(description='MESA2 Training')

# ------------- train setting ----------------------------------------------------
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--epochs', '-ep', default=200, type=int, metavar='N',
                    help='number of total initial epochs to run (default: 200)')
parser.add_argument('--qauntize_epochs', '-qep', default=10, type=int, metavar='N',
                    help='number of quantize retrain epochs to run (default: 10)')
parser.add_argument('--reepochs', '-reep', default=40, type=int, metavar='N',
                    help='number of pruning retrain epochs to run (default: 40)')
parser.add_argument('--prune-interval', '-pi', default=1, type=int, metavar='N',
                    help='prune interval when using prune-mode "filter-gm" (default: 1)')
parser.add_argument('--start-epoch', '-sep', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-bs', default=256, type=int,metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate','-lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-pf', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# ------------- alpha and beta ----------------------------------------------------
parser.add_argument('--alpha', '-al', default=0.1, type=float, metavar='M',
                    help='alpha(default=0.1)')
parser.add_argument('--beta', '-be', default=0.1, type=float, metavar='M',
                    help='beta(default=0.1)')

# ---------- pruning rate for conv1, conv2, conv3, conv4, conv5 --------------------
parser.add_argument('--prune-rates', nargs='+', type=float,
                    default=[0.16, 0.62, 0.65, 0.63, 0.63],
                    help='pruning rate for AlexNet conv layer ' +
                         '(default=[0.16, 0.62, 0.65, 0.63, 0.63])')

# ---------- partition size for fc1, fc2, fc3 --------------------------------------
parser.add_argument("--partition", '-p', dest="partition",
                    action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='partition size of fc layer (eg. fc1=8,fc2=8,fc3=10)')

# ---------- quantize bits ---------------------------------------------------------
parser.add_argument('--bits', '-b', default=5, type=int,
                    help='quantizatize bit(default=5)')

# ---------- log file --------------------------------------------------------------
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')

# ---------- base dir --------------------------------------------------------------
parser.add_argument('--save-dir', '-sd', type=str,default='model_default',
                    help='model store path(defalut="model_default")')

# ---------- load file -------------------------------------------------------------
parser.add_argument('--load-model', '-lm', type=str,default='checkpoint_default.tar',
                    help='load exist checkpoint file')

# ---------- train process ------------------------------------------------------------
parser.add_argument('--initial-process', '-initp', dest='initial_process', action='store_true',
                    help='run initial process or not')
parser.add_argument('--pruning-process', '-prunep', dest='pruning_process', action='store_true',
                    help='run pruning process or not')
parser.add_argument('--qunatize-process', '-quanp', dest='quantize_process', action='store_true',
                    help='run quantize process or not')

# ---------- prune mode -----------------------------------------------------
parser.add_argument('--prune-mode', '-pm', default='filter-norm', type=str, metavar='M',
                    help='filter: pruned by filter percentile: pruned by percentile channel: pruned by channel\n')

# ---------- dnn or cnn or all -----------------------------------------------------
parser.add_argument('--model-mode', '-mm', default='c', type=str, metavar='M',
                    help='d:only qauntize dnn c:only qauntize cnn a:all qauntize\n')
parser.add_argument('--fc-mask', '-fc-mask', dest='fc_mask', action='store_true',
                    help='use fully connected layer mask or not')

# ---------- save folders -----------------------------------------------------
parser.add_argument('--out-oldweight-folder', default='model_before_prune', type=str,
                    help='path to model output')
parser.add_argument('--out-pruned-folder', default='model_prune', type=str,
                    help='path to model output')
parser.add_argument('--out-pruned-re-folder', default='model_prune_re', type=str,
                    help='path to model output')
parser.add_argument('--out-quantized-folder', default='model_quantized', type=str,
                    help='path to model output')
parser.add_argument('--out-quantized-re-folder', default='model_quantized_retrain', type=str,
                    help='path to model output')

args = train_loader = val_loader = None


def environ_setting():
    global args, train_loader, val_loader
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings('ignore')
    args = parser.parse_args()
    args.method_str = util.get_method_str(args)
    train_loader, val_loader = dataSet.get_cifar100_dataSet(args)
    args.best_prec1 = 0.0
    torch.manual_seed(args.seed)
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using CUDA!")
        args.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print('Not using CUDA!')
        args.device = torch.device("cpu")


def check_folders_exist():
    os.makedirs(f'{args.save_dir}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_oldweight_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_pruned_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_pruned_re_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_quantized_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_quantized_re_folder}', exist_ok=True)
    util.log(f"{args.save_dir}/{args.log}", "--------------------------configure----------------------")
    util.log(f"{args.save_dir}/{args.log}", f"{args}\n")


def run_process():
    print(f'Method: {args.method_str}')  # alpha corresponds to fc, beta corresponds to conv
    if args.fc_mask:
        model = AlexNet_mask.AlexNet_mask(args.partition, mask_flag=True).to(args.device)
    else:
        model = AlexNet_mask.AlexNet(mask_flag=True).to(args.device)
    if os.path.isfile(f"{args.load_model}"):
        model, args.best_prec1 = util.load_checkpoint(model, f"{args.load_model}", args)
        print("-------load " + f"{args.load_model} ({args.best_prec1:.3f})----")
    if args.initial_process:
        model = initial_process(model)
    if args.pruning_process:
        model = pruning_process(model)
    if args.quantize_process:
        model = quantize_process(model)


def initial_process(model):
    print(model)
    util.print_model_parameters(model)
    print("------------------------- Initial training -------------------------------")
    model = util.initial_train(model, args, train_loader, val_loader, 'initial')
    accuracy, accuracy5 = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(f"{args.save_dir}/{args.log}", f"initial_accuracy\t{accuracy} ({accuracy5})")
    return model


def pruning_process(model):
    print("------------------------- Before pruning --------------------------------")
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    accuracy, accuracy5 = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(f"{args.save_dir}/{args.log}", f"before pruning accuracy\t{accuracy} ({accuracy5})")

    print("------------------------- pruning CNN--------------------------------------")
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
        model.set_conv_prune_indice_dict(mode=args.prune_mode)

    print("------------------------------- After prune CNN ----------------------------")
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    accuracy, accuracy5 = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(f"{args.save_dir}/{args.log}", f"after pruning accuracy\t{accuracy} ({accuracy5})")

    print("------------------------- start retrain after prune CNN----------------------------")
    util.initial_train(model, args, train_loader, val_loader, 'prune_retrain')

    print("------------------------- After Retraining -----------------------------")
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    accuracy, accuracy5 = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(f"{args.save_dir}/{args.log}", f"after pruning and retrain accuracy\t{accuracy} ({accuracy5})")

    return model


def quantize_process(model):
    print('------------------------------- accuracy before weight sharing ----------------------------------')
    accuracy, accuracy5 = util.validate(val_loader, model, args, topk=(1, 5))
    util.log(f"{args.save_dir}/{args.log}", f"accuracy before weight sharing\t{accuracy} ({accuracy5})")

    print('------------------------------- accuacy after weight sharing -------------------------------')
    old_weight_list, new_weight_list, quantized_index_list, quantized_center_list = (
        apply_weight_sharing(model, args.model_mode, args.device, args.bits))
    accuracy, accuracy5 = util.validate(val_loader, model, args, topk=(1, 5))
    util.save_checkpoint({
        'state_dict': model.state_dict(),
        'best_prec1': accuracy,
    }, file_path=os.path.join(args.save_dir, f'checkpoint_quantized_{args.method_str}_initial.tar'))
    util.log(f"{args.save_dir}/{args.log}", f"accuracy after weight sharing {args.bits}bits\t{accuracy} ({accuracy5})")

    print('------------------------------- retraining -------------------------------------------')
    util.quantized_retrain(model, args, quantized_index_list, quantized_center_list, train_loader, val_loader)
    accuracy, accuracy5 = util.validate(val_loader, model, args, topk=(1, 5))
    util.save_checkpoint({
       'state_dict': model.state_dict(),
       'best_prec1': accuracy,
    }, file_path=os.path.join(args.save_dir, f'checkpoint_quantized_{args.method_str}_end.tar'))
    util.log(f"{args.save_dir}/{args.log}", f"acc after qauntize and retrain\t{accuracy} ({accuracy5})")

    return model


def main():
    environ_setting()
    check_folders_exist()
    run_process()


if __name__ == '__main__':
    main()
