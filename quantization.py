import torch
import numpy as np
from sklearn.cluster import KMeans


def apply_weight_sharing(model, args):
    quan_name2labels = dict()
    for name, param in model.named_parameters():
        if (args.model_mode == 'd' and 'conv' in name or
                args.model_mode == 'c' and 'fc' in name or
                'mask' in name or
                'bias' in name or
                'bn' in name):
            print(f'{name:20} | {str(param.size()):35} | => pass')
            continue

        ori_weights = param.data.cpu().numpy()
        if len(ori_weights.shape) == 4 and args.model_mode != 'd':  # convolution layer
            quan_range = 2 ** int(args.bits['conv'])
        elif len(ori_weights.shape) == 2 and args.model_mode != 'c':  # dense layer
            quan_range = 2 ** int(args.bits['fc'])
        else:
            raise Exception
        print(f'{name:20} | {str(param.size()):35} | => quantize to {quan_range} indices')
        nonzero_flat_weights = ori_weights[ori_weights != 0].reshape(-1, 1)
        nonzero_indice = np.where(ori_weights != 0)
        space = np.linspace(np.min(nonzero_flat_weights), np.max(nonzero_flat_weights), num=quan_range).reshape(-1, 1)
        kmeans = KMeans(n_clusters=len(space), init=space, n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(nonzero_flat_weights)

        # reconstruct quantized weights and set model.param
        quan_weights = np.zeros(ori_weights.shape)
        quan_weights[nonzero_indice] = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        param.data = torch.from_numpy(quan_weights).float().to(args.device)

        # reconstruct quantized labels
        quan_labels = np.zeros(ori_weights.shape)
        quan_labels[nonzero_indice] = kmeans.labels_
        quan_name2labels[name] = quan_labels.astype(int)

    return quan_name2labels

