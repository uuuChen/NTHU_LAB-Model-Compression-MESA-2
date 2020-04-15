import torch
import numpy as np
from sklearn.cluster import KMeans
import scipy.linalg


def apply_weight_sharing(model, model_mode, device, bits=4):
    old_weight_list = list()
    new_weight_list = list()
    quantized_index_list = list()
    quantized_center_list = list()
    new_weight = quantized_index = quantized_center = None
    for name, module in model.named_children():
        print(name, module)
        old_weight = module.weight.data.cpu().numpy()
        shape = old_weight.shape
        if len(shape) == 1:  # bn layer
            continue
        elif len(shape) > 2:  # convolution layer
            if model_mode == 'd':  # skip convolution layer
                print('\tpass')
                continue
            weight = old_weight.reshape(1, -1)
            zero_index = np.where(weight == 0)[1]
            mat = weight[0]
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True,
                            algorithm="full")
            kmeans.fit(mat.reshape(-1, 1))
            cluster_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            cluster_weight[zero_index] = 0.0
            new_weight = cluster_weight
            new_weight = new_weight.reshape(shape)
            module.weight.data = torch.from_numpy(new_weight).float().to(device)
            quantized_index = kmeans.labels_.reshape(shape)
            quantized_center = kmeans.cluster_centers_
        elif len(shape) == 2:  # dense layer
            if model_mode == 'c':
                print('\tpass')
                continue
            partition_num = int(model.partition_size[name])
            N = int(old_weight.shape[0] / partition_num)
            M = int(old_weight.shape[1] / partition_num)
            print('\tpartition number:', partition_num)
            print('\trow number/partition:', N)
            print('\tcol number/partition:', M)
            block_list = list()
            j = 0
            for i in range(partition_num):
                block_list.append(old_weight[i*N:(i+1)*N, j*M:(j+1)*M])
                j += 1
            blocks = np.array(block_list)
            mat = blocks.reshape(1, -1)[0]
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True,
                            algorithm="full")
            kmeans.fit(mat.reshape(-1, 1))
            cluster_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)

            t = tuple(cluster_weight.reshape(blocks.shape))
            cluster_weight_arr = scipy.linalg.block_diag(*t)
            new_weight = np.zeros(shape)
            new_weight[:cluster_weight_arr.shape[0], :cluster_weight_arr.shape[1]] = cluster_weight_arr
            new_weight = new_weight.reshape(shape)  # seems useless
            module.weight.data = torch.from_numpy(new_weight).float().to(device)

            index_t = tuple(kmeans.labels_.reshape(blocks.shape))
            index_arr = scipy.linalg.block_diag(*index_t)
            quantized_index = np.zeros(shape)
            quantized_index[:index_arr.shape[0], :index_arr.shape[1]] = index_arr
            quantized_center = kmeans.cluster_centers_

        old_weight_list.append(old_weight)
        new_weight_list.append(new_weight)
        quantized_index_list.append(quantized_index.astype(int))
        quantized_center_list.append(quantized_center)

    return old_weight_list, new_weight_list, quantized_index_list, quantized_center_list


