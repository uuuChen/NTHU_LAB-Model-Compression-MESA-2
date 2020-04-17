import torch
import numpy as np
from sklearn.cluster import KMeans
import scipy.linalg


def apply_weight_sharing(model, args):
    quan_labels_list = list()
    quan_labels = None

    for name, param in model.named_parameters():
        if (args.model_mode == 'd' and 'conv' in name or
                args.model_mode == 'c' and 'fc' in name or
                'mask' in name or
                'bias' in name or
                'bn' in name):
            print(f'{name:15} | {str(param.size()):35} | => pass')
            continue

        ori_weights = param.data.cpu().numpy()
        ori_shape = ori_weights.shape
        if len(ori_shape) == 4 and args.model_mode != 'd':  # convolution layer
            quan_range = 2**int(args.bits['conv'])
            print(f'{name:15} | {str(param.size()):35} | => quantize to {quan_range} indice')
            nonzero_flat_weights = ori_weights[ori_weights != 0].reshape(-1, 1)
            nonzero_indice = np.where(ori_weights != 0)
            space = np.linspace(
                np.min(nonzero_flat_weights), np.max(nonzero_flat_weights), num=2).reshape(-1, 1)
            kmeans = KMeans(n_clusters=len(space), init=space, n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(nonzero_flat_weights)

            # reconstruct quantized weights and set model.param
            quan_weights = np.zeros(ori_weights.shape)
            quan_weights[nonzero_indice] = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            param.data = torch.from_numpy(quan_weights).float().to(args.device)

            # reconstruct quantized labels
            quan_labels = np.zeros(ori_weights.shape)
            quan_labels[nonzero_indice] = kmeans.labels_

        elif len(ori_shape) == 2 and args.model_mode != 'c':  # dense layer
            quan_range = 2 ** int(args.bits['fc'])
            partition_num = int(model.partition_size[name[:3]])
            block_rows = ori_weights.shape[0] // partition_num
            block_cols = ori_weights.shape[1] // partition_num
            print(f'{name:15} | {str(param.shape):35} | partition: {partition_num} , block_rows: {block_rows} , '
                  f'block_cols: {block_cols} | => quantize to {quan_range} indice')
            blocks = list()
            for i in range(partition_num):
                block = ori_weights[
                    i * block_rows: (i + 1) * block_rows,
                    i * block_cols: (i + 1) * block_cols
                ]
                blocks.append(block)
            blocks = np.array(blocks)
            nonzero_flat_weights = blocks.reshape(-1, 1)
            space = np.linspace(
                np.min(nonzero_flat_weights), np.max(nonzero_flat_weights), num=quan_range).reshape(-1, 1)
            kmeans = KMeans(n_clusters=len(space), init=space, n_init=1, precompute_distances=True,  algorithm="full")
            kmeans.fit(nonzero_flat_weights)

            # reconstruct quantized weights and set model.param
            nonzero_quan_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(blocks.shape)
            blocks_quan_weights = scipy.linalg.block_diag(*tuple(nonzero_quan_weights))
            quan_weights = np.zeros(ori_shape)
            quan_weights[:blocks_quan_weights.shape[0], :blocks_quan_weights.shape[1]] = blocks_quan_weights
            param.data = torch.from_numpy(quan_weights).float().to(args.device)

            # reconstruct quantized labels
            blocks_quan_labels = scipy.linalg.block_diag(*tuple(kmeans.labels_.reshape(blocks.shape)))
            quan_labels = np.zeros(ori_shape)
            quan_labels[:blocks_quan_labels.shape[0], :blocks_quan_labels.shape[1]] = blocks_quan_labels

        quan_labels_list.append(quan_labels.astype(int))

    return quan_labels_list


