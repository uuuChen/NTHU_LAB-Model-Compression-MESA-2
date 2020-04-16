import torch
import numpy as np
from sklearn.cluster import KMeans
import scipy.linalg


def apply_weight_sharing(model, model_mode, device, bits):
    ori_weights_list = list()
    quan_weights_list = list()
    quan_labels_list = list()
    quan_centers_list = list()
    ori_weights = quan_weights = quan_labels = quan_centers = None
    for name, param in model.named_parameters():
        if (model_mode == 'd' and 'conv' in name or
                model_mode == 'c' and 'fc' in name or
                'mask' in name or
                'bias' in name or
                'bn' in name):
            print(f'{name:15} | {str(param.size()):35} | => pass')
            continue

        ori_weights = param.data.cpu().numpy()
        ori_shape = ori_weights.shape

        if len(ori_shape) == 4 and model_mode != 'd':  # convolution layer
            print(f'{name:15} | {str(param.size()):35} | => quantize to {2**int(bits["conv"])} indice')
            flat_weights = ori_weights.reshape(-1, 1)
            zero_index = np.where(flat_weights == 0)[0]
            space = np.linspace(np.min(flat_weights), np.max(flat_weights), num=2**int(bits['conv'])).reshape(-1, 1)
            kmeans = KMeans(n_clusters=len(space), init=space, n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(flat_weights)
            quan_weights = kmeans.cluster_centers_[kmeans.labels_]
            quan_weights[zero_index] = 0.0
            param.data = torch.from_numpy(quan_weights.reshape(ori_shape)).float().to(device)
            quan_labels = kmeans.labels_.reshape(ori_shape)
            quan_centers = kmeans.cluster_centers_

        elif len(ori_shape) == 2 and model_mode != 'c':  # dense layer
            partition_num = int(model.partition_size[name])
            block_rows = ori_weights.shape[0] // partition_num
            block_cols = ori_weights.shape[1] // partition_num
            print(f'{name:15} | {param.shape:35} | partition: {partition_num:15}  block_rows: {block_rows:15} '
                  f' {block_cols:15} | => quantize to {2**int(bits["fc"])} indice')
            blocks = list()
            for i in range(partition_num):
                block = ori_weights[
                    i * block_rows: (i + 1) * block_rows,
                    i * block_cols: (i + 1) * block_cols
                ]
                blocks.append(block)
            blocks = np.array(blocks)
            flat_weights = blocks.reshape(-1, 1)
            space = np.linspace(np.min(flat_weights), np.max(flat_weights), num=2**int(bits['fc'])).reshape(-1, 1)
            kmeans = KMeans(n_clusters=len(space), init=space, n_init=1, precompute_distances=True,  algorithm="full")
            kmeans.fit(flat_weights)

            # get quantized weights
            nonzero_quan_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(blocks.shape)
            blocks_quan_weights = scipy.linalg.block_diag(*tuple(nonzero_quan_weights))
            quan_weights = np.zeros(ori_shape)
            quan_weights[:blocks_quan_weights.shape[0], :blocks_quan_weights.shape[1]] = blocks_quan_weights
            param.data = torch.from_numpy(quan_weights).float().to(device)

            # get quantized labels
            blocks_quan_labels = scipy.linalg.block_diag(*tuple(kmeans.labels_.reshape(blocks.shape)))
            quan_labels = np.zeros(ori_shape)
            quan_labels[:blocks_quan_labels.shape[0], :blocks_quan_labels.shape[1]] = blocks_quan_labels

            # get quantized centers (values)
            quan_centers = kmeans.cluster_centers_

        ori_weights_list.append(ori_weights)
        quan_weights_list.append(quan_weights)
        quan_labels_list.append(quan_labels.astype(int))
        quan_centers_list.append(quan_centers)

    return ori_weights_list, quan_weights_list, quan_labels_list, quan_centers_list


