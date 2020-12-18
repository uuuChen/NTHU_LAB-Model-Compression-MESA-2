import numpy as np
import util
from huffmancoding import huffman_encode


def to_indices(layer):
    value_set = np.unique(layer)
    value2idx_dict = dict(zip(value_set, range(len(value_set))))
    return np.vectorize(value2idx_dict.get)(layer)


def cycle_distance(a, b, max_distance):
    if b - a >= 0:
        return b - a
    else:
        return b - a + max_distance


def get_index_percentage(indices_arr, index):
    return len(np.where(indices_arr == index)[0]) / len(indices_arr.reshape(-1)) * 100.


def matrix_cycledistance(array_x, array_y, max_distance):
    flat_list_x = list(array_x.reshape(-1))
    flat_list_y = list(array_y.reshape(-1))
    distance_array = list()
    for x_val, y_val in list(zip(flat_list_x, flat_list_y)):
        distance_array.append(cycle_distance(x_val, y_val, max_distance))
    distance_array = np.array(distance_array).reshape(array_x.shape)
    return distance_array


def conv_filter3d_delta_coding(conv4d_indices, args, max_distance, pruned_filter_indices, pruned_channel_indices, directory):
    first_filter = prev_filter = None
    delta_filters = list()
    for i, cur_filter in list(enumerate(conv4d_indices)):
        if i == 0:
            first_filter = cur_filter
        else:
            delta_filters.append(matrix_cycledistance(prev_filter, cur_filter, max_distance))
        prev_filter = cur_filter
    delta_filters = np.array(delta_filters)

    # print quantized indices statistics
    for i in range(2 ** int(args.bits['conv']) + 1):
        print(f'{i}: {get_index_percentage(conv4d_indices, i) :.2f} % | {get_index_percentage(delta_filters, i) :.2f} %')

    # Encode
    t0, d0 = huffman_encode(first_filter.astype('float32'), directory)
    t1, d1 = huffman_encode(delta_filters.astype('float32'), directory)
    t2, d2 = huffman_encode(pruned_filter_indices.astype('int32'), directory)
    t3, d3 = huffman_encode(pruned_channel_indices.astype('int32'), directory)

    # Get statistics
    original = first_filter.nbytes + delta_filters.nbytes + pruned_filter_indices.nbytes + pruned_channel_indices.nbytes
    compressed = t0 + t1 + t2 + t3 + d0 + d1 + d2 + d3

    return original, compressed


def conv_part_filter3d_delta_coding(conv4d_indices, args, max_distance, pruned_filter_indices, pruned_channel_indices, directory):
    # Sorted by absolute sum
    sum_of_filters = np.sum(np.abs(conv4d_indices.reshape(conv4d_indices.shape[0], -1)), 1)
    part_filters_num = 10  # hyperparameter
    sorted_filters_indices = np.argsort(sum_of_filters)
    part_filters_indices, left_filters_indices = sorted_filters_indices[:part_filters_num], sorted_filters_indices[part_filters_num:]
    part_conv4d_indices, left_conv4d_indices = conv4d_indices[part_filters_indices, :, :, :], conv4d_indices[left_filters_indices, :, :, :]

    original = compressed = 0
    for indices4d in [part_conv4d_indices, left_conv4d_indices]:
        first_filter = prev_filter = None
        delta_filters = list()
        for i, cur_filter in list(enumerate(indices4d)):
            if i == 0:
                first_filter = cur_filter
            else:
                delta_filters.append(matrix_cycledistance(prev_filter, cur_filter, max_distance))
            prev_filter = cur_filter
        delta_filters = np.array(delta_filters)

        # print quantized indices statistics
        for i in range(2 ** int(args.bits['conv']) + 1):
            print(f'{i}: {get_index_percentage(indices4d, i) :.2f} % | {get_index_percentage(delta_filters, i) :.2f} %')

        # Encode
        t0, d0 = huffman_encode(first_filter.astype('float32'), directory)
        t1, d1 = huffman_encode(delta_filters.astype('float32'), directory)
        original += (first_filter.nbytes + delta_filters.nbytes)
        compressed += (t0 + t1 + d0 + d1)

    # Encode
    t2, d2 = huffman_encode(pruned_filter_indices.astype('int32'), directory)
    t3, d3 = huffman_encode(pruned_channel_indices.astype('int32'), directory)

    # Get statistics
    original += (pruned_filter_indices.nbytes + pruned_channel_indices.nbytes)
    compressed += (t2 + t3 + d2 + d3)

    return original, compressed


def conv_width1d_delta_coding(conv4d_indices, args, max_distance, pruned_filter_indices, pruned_channel_indices, directory):
    indices2d = conv4d_indices.reshape(-1, conv4d_indices.shape[3])  # shape: (kn * cn * h, w)
    first_col = prev_col = None
    delta_cols = list()
    for i in range(indices2d.shape[1]):
        cur_col = indices2d[:, i]
        if i == 0:
            first_col = cur_col
        else:
            delta_cols.append(matrix_cycledistance(prev_col, cur_col, max_distance))
        prev_col = cur_col
    delta_cols = np.array(delta_cols)

    # print quantized indices statistics
    for i in range(2 ** int(args.bits['conv']) + 1):
        print(
            f'{i}: {get_index_percentage(conv4d_indices, i) :.2f} % | {get_index_percentage(delta_cols, i) :.2f} %')

    # Encode
    t0, d0 = huffman_encode(first_col.astype('float32'), directory)
    t1, d1 = huffman_encode(delta_cols.astype('float32'), directory)
    t2, d2 = huffman_encode(pruned_filter_indices.astype('int32'), directory)
    t3, d3 = huffman_encode(pruned_channel_indices.astype('int32'), directory)

    # Get statistics
    original = first_col.nbytes + delta_cols.nbytes + pruned_filter_indices.nbytes + pruned_channel_indices.nbytes
    compressed = t0 + t1 + t2 + t3 + d0 + d1 + d2 + d3

    return original, compressed


def conv_non_delta_coding(unpruned_conv_indices, pruned_filter_indices, pruned_channel_indices, directory):
    # Encode
    t0, d0 = huffman_encode(unpruned_conv_indices.astype('float32'), directory)
    t1, d1 = huffman_encode(pruned_filter_indices.astype('int32'), directory)
    t2, d2 = huffman_encode(pruned_channel_indices.astype('int32'), directory)

    # Get statistics
    original = unpruned_conv_indices.nbytes + pruned_filter_indices.nbytes + pruned_channel_indices.nbytes
    compressed = t0 + t1 + t2 + d0 + d1 + d2

    return original, compressed


def get_fc_delta_blocks(fc2d_indices, fc2d_arr, partition_size, max_distance):
    num_of_block_rows = fc2d_indices.shape[0] // partition_size
    num_of_block_cols = fc2d_indices.shape[1] // partition_size
    delta_blocks = list()
    first_block = prev_block = None
    for i in range(partition_size):
        if i == 0:
            first_block = fc2d_indices[:num_of_block_rows, :num_of_block_cols]
            prev_block = first_block
        else:
            cur_block = fc2d_indices[
                i*num_of_block_rows: (i+1)*num_of_block_rows,
                i*num_of_block_cols: (i+1)*num_of_block_cols
            ]
            delta_blocks.append(matrix_cycledistance(prev_block, cur_block, max_distance))
            prev_block = cur_block
    first_block = first_block.astype('float32')
    delta_blocks = np.array(delta_blocks).astype('float32')

    # print quantized indices statistics
    nonzero_indices = np.append(first_block.reshape(-1), delta_blocks.reshape(-1))
    for i in range(2 ** 5 + 1):
        print(f'{i}: {get_index_percentage(nonzero_indices, i) :.2f} % | {get_index_percentage(delta_blocks, i) :.2f} %')

    return first_block, delta_blocks


def mesa2_huffman_encode_conv4d(model, name, args, conv4d_arr, max_distance, directory):
    if model.convLayerName2pruneIndices is None:
        model.set_conv_indices_dict()
    pruned_filter_indices, pruned_channel_indices = model.convLayerName2pruneIndices[name]
    unpruned_conv_layer_indices = util.get_unpruned_conv_layer_weights(to_indices(conv4d_arr), model, name)
    if args.conv_loss_func == "filter3d-delta":
        original, compressed = conv_filter3d_delta_coding(unpruned_conv_layer_indices, args, max_distance, pruned_filter_indices, pruned_channel_indices, directory)
    elif args.conv_loss_func == "part-filter3d-delta":
        original, compressed = conv_part_filter3d_delta_coding(unpruned_conv_layer_indices, args, max_distance, pruned_filter_indices, pruned_channel_indices, directory)
    elif args.conv_loss_func == "width1d-delta":
        original, compressed = conv_width1d_delta_coding(unpruned_conv_layer_indices, args, max_distance, pruned_filter_indices, pruned_channel_indices, directory)
    else:
        original, compressed = conv_non_delta_coding(unpruned_conv_layer_indices, pruned_filter_indices, pruned_channel_indices, directory)
    return original, compressed


def mesa2_huffman_encode_fc2d(fc2d_arr, partitionsize, max_distance, directory):
    first_block_arr, delta_blocks_arr = get_fc_delta_blocks(to_indices(fc2d_arr), fc2d_arr, partitionsize, max_distance)

    # Encode
    t0, d0 = huffman_encode(first_block_arr, directory)
    t1, d1 = huffman_encode(delta_blocks_arr, directory)

    # Get statistics
    original = first_block_arr.nbytes + delta_blocks_arr.nbytes
    compressed = t0 + t1 + d0 + d1

    return original, compressed


def mesa2_huffman_encode_model(model, args, directory='encodings/'):
    original_total = compressed_total = 0  # the unit is bytes
    print(f"{'Layer':<15} | {'original':>10} {'compressed':>10} {'improvement':>11} {'percent':>7}")
    print('-' * 70)
    for name, param in model.named_parameters():
        if 'mask' in name or 'bn' in name:
            continue
        original = compressed = 0
        ignore = False
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            if 'conv' in name and args.model_mode != 'd':
                original, compressed = mesa2_huffman_encode_conv4d(model, name.split('.')[0], args, weight, 2**int(args.bits['conv']), directory)
            elif 'fc' in name and args.model_mode != 'c':
                original, compressed = mesa2_huffman_encode_fc2d(weight, int(args.partition[name[0:3]]), 2**int(args.bits['fc']), directory)
            else:
                ignore = True
        else:  # bias
            if 'conv' in name and args.model_mode != 'd' or 'fc' in name and args.model_mode != 'c':
                bias = param.data.cpu().numpy()
                original = compressed = bias.nbytes
            else:
                ignore = True

        # Print and log statistics
        if ignore:
            log_str = f"{name:<15} | {'* pass':>10}"
            util.log(args.log_file_path, log_str)
            print(log_str)
        else:
            original_total += original
            compressed_total += compressed
            log_str = f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%"
            util.log(args.log_file_path, log_str)
            print(log_str)

    # Print and log statistics
    part_model_original_bytes = util.get_part_model_original_bytes(model, args)
    log_str = (f'\ncompression rate (without pruned params): {original_total} / {compressed_total} ({ original_total / compressed_total:.3f} X) \n' 
               f'compression rate (include pruned params): { part_model_original_bytes} / {compressed_total} ({ part_model_original_bytes / compressed_total:.3f} X)')
    util.log(args.log_file_path, log_str)
    print('-' * 70)
    print(log_str)
