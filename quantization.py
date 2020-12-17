import torch
import numpy as np
from sklearn.cluster import KMeans
import util


def apply_weight_sharing(model, args):
    layerName2quanIndices = dict()
    for name, param in model.named_parameters():
        if util.be_ignored(name, args.model_mode):
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
        nonzero_indices = np.where(ori_weights != 0)
        space = np.linspace(np.min(nonzero_flat_weights), np.max(nonzero_flat_weights), num=quan_range).reshape(-1, 1)
        kmeans = KMeans(n_clusters=len(space), init=space, n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(nonzero_flat_weights)

        # reconstruct quantized weights and set model.param
        quan_weights = np.zeros(ori_weights.shape)
        quan_weights[nonzero_indices] = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        param.data = torch.from_numpy(quan_weights).float().to(args.device)

        # reconstruct quantized labels
        quan_indices = np.zeros(ori_weights.shape)
        quan_indices[nonzero_indices] = kmeans.labels_
        layerName2quanIndices[name] = quan_indices.astype(int)

    return layerName2quanIndices

