import numpy as np
from numpy import array
from huffmancoding import huffman_encode


def get_model_origin_bytes(model, args):
    model_params = 0
    for name, param in model.named_parameters():
        if 'mask' in name or 'bn' in name or 'bias' in name:
            continue
        if 'conv' in name and args.model_mode != 'd':
            model_params += len(param.reshape(-1))
        if 'fc' in name and args.model_mode != 'c':
            model_params += len(param.reshape(-1))
    model_bytes = model_params * 4
    return model_bytes


def to_index(layer):
    value_set = np.unique(layer)
    value2idx_dict = dict(zip(value_set, range(len(value_set))))
    return np.vectorize(value2idx_dict.get)(layer)


def cycledistance(a, b, maxdistance):
    if b - a >= 0:
        return b - a
    else:
        return b - a + maxdistance


def matrix_cycledistance(array_x, array_y, maxdistance):
    flat_list_x = list(array_x.reshape(-1))
    flat_list_y = list(array_y.reshape(-1))
    distancearray = list()
    for x_val, y_val in list(zip(flat_list_x, flat_list_y)):
        distancearray.append(cycledistance(x_val, y_val, maxdistance))
    distancearray = np.array(distancearray).reshape(array_x.shape)
    return distancearray


def delta_encoding_conv4d(conv4d_arr, maxdistance):
    pruned_filter_indice = np.where(np.sum(conv4d_arr.reshape(conv4d_arr.shape[0], -1), axis=1) == 0)[0]
    left_filter_indice = np.array(list(set(range(conv4d_arr.shape[0])).difference(pruned_filter_indice)))
    conv4d_index = to_index(conv4d_arr)
    left_filters_index = conv4d_index[left_filter_indice, :, :, :]
    first_filter = prev_filter = None
    delta_filters = list()
    for i, cur_filter in list(enumerate(left_filters_index)):
        if i == 0:
            first_filter = cur_filter
        else:
            delta_filters.append(matrix_cycledistance(prev_filter, cur_filter, maxdistance))
        prev_filter = cur_filter
    first_filter = first_filter.astype('float32')
    delta_filters = np.array(delta_filters).astype('float32')
    pruned_filter_indice = pruned_filter_indice.astype('int32')
    return first_filter, delta_filters, pruned_filter_indice


def getblocks(fc, x_axis, y_axis, partitionsize):
    block_histogram = list()
    for i in range(partitionsize):
        blocks = list()
        for j in range(x_axis // partitionsize):
            for k in range(y_axis // partitionsize):
                blocks.append(fc[j][k])
        if i == 0:
            block_histogram = array(blocks)
        else:
            block_histogram = np.concatenate((block_histogram, array(blocks)), axis=0)
    return block_histogram


def delta_encoding_fc2d(fc2d_arr, partitionsize, maxdistance):
    fc2d_index = to_index(fc2d_arr)
    num_of_block_rows = fc2d_index.shape[0] // partitionsize
    num_of_block_cols = fc2d_index.shape[1] // partitionsize
    delta_blocks = list()
    first_block = prev_block = None
    for i in range(partitionsize):
        if i == 0:
            first_block = fc2d_index[:num_of_block_rows, :num_of_block_cols]
            prev_block = first_block
        else:
            cur_block = fc2d_index[
                i*num_of_block_rows: (i+1)*num_of_block_rows,
                i*num_of_block_cols: (i+1)*num_of_block_cols
            ]
            delta_blocks.append(matrix_cycledistance(prev_block, cur_block, maxdistance))
            prev_block = cur_block
    first_block = first_block.astype('float32')
    delta_blocks = np.array(delta_blocks).astype('float32')
    return first_block, delta_blocks


def mesa2_huffman_encode_conv4d(conv4d_arr, maxdistance, directory):
    first_filter_arr, delta_filters_arr, pruned_filter_indice = delta_encoding_conv4d(conv4d_arr, maxdistance)

    # Encode
    t0, d0 = huffman_encode(first_filter_arr, directory)
    t1, d1 = huffman_encode(delta_filters_arr, directory)
    t2, d2 = huffman_encode(pruned_filter_indice, directory)

    # Print statistics
    original = first_filter_arr.nbytes + delta_filters_arr.nbytes + pruned_filter_indice.nbytes
    compressed = t0 + t1 + t2 + d0 + d1 + d2

    return original, compressed


def mesa2_huffman_encode_fc2d(fc2d, partitionsize, maxdistance, directory):
    fc2d = to_index(fc2d)
    first_block_arr, delta_blocks_arr = delta_encoding_fc2d(fc2d, partitionsize, maxdistance)

    # Encode
    t0, d0 = huffman_encode(first_block_arr, directory)
    t1, d1 = huffman_encode(delta_blocks_arr, directory)

    # Print statistics
    original = first_block_arr.nbytes + delta_blocks_arr.nbytes
    compressed = t0 + t1 + d0 + d1

    return original, compressed


def mesa2_huffman_encode_model(model, args, directory='encodings/'):
    original_total = compressed_total = 0  # the unit is bytes
    for name, param in model.named_parameters():
        if 'mask' in name or 'bn' in name or 'bias' in name:
            continue
        param_arr = param.data.cpu().numpy()
        original = compressed = 0
        if 'conv' in name and args.model_mode is not 'd':
            original, compressed = mesa2_huffman_encode_conv4d(param_arr, 2**int(args.bits['conv']), directory)
        if 'fc' in name and args.model_mode is not 'c':
            original, compressed = mesa2_huffman_encode_fc2d(
                param_arr, int(args.partition[name[0:3]]), 2**int(args.bits['fc']), directory)
        original_total += original
        compressed_total += compressed
    ori_model_bytes = get_model_origin_bytes(model, args)
    print(f'compression rate: {ori_model_bytes/compressed_total:.3f} X')



