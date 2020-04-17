import numpy as np
from huffmancoding import huffman_encode
import util


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
    t2, d2 = huffman_encode(pruned_filter_indice, directory) if len(pruned_filter_indice) != 0 else (0, 0)

    # Print statistics
    original = first_filter_arr.nbytes + delta_filters_arr.nbytes + pruned_filter_indice.nbytes
    compressed = t0 + t1 + t2 + d0 + d1 + d2

    return original, compressed


def mesa2_huffman_encode_fc2d(fc2d_arr, partitionsize, maxdistance, directory):
    fc2d_index = to_index(fc2d_arr)
    first_block_arr, delta_blocks_arr = delta_encoding_fc2d(fc2d_index, partitionsize, maxdistance)

    # Encode
    t0, d0 = huffman_encode(first_block_arr, directory)
    t1, d1 = huffman_encode(delta_blocks_arr, directory)

    # Print statistics
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
                original, compressed = mesa2_huffman_encode_conv4d(weight, 2**int(args.bits['conv']), directory)
            elif 'fc' in name and args.model_mode != 'c':
                original, compressed = mesa2_huffman_encode_fc2d(
                    weight, int(args.partition[name[0:3]]), 2**int(args.bits['fc']), directory)
            else:
                ignore = True
        else:  # bias
            if ('conv' in name and args.model_mode != 'd' or
                    'fc' in name and args.model_mode != 'c'):
                bias = param.data.cpu().numpy()
                original = bias.nbytes
                compressed = original
            else:
                ignore = True

        # Print and log statistics
        if ignore:
            log_str = f"{name:<15} | {'* pass':>10}"
            print(log_str)
            util.log(f"{args.save_dir}/{args.log}", log_str)
        else:
            original_total += original
            compressed_total += compressed
            log_str = (f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x " 
                       f"{100 * compressed / original:>6.2f}%")
            print(log_str)
            util.log(f"{args.save_dir}/{args.log}", log_str)

    # Print and log statistics
    part_model_original_bytes = util.get_part_model_original_bytes(model, args)
    log_str = (f'\ncompression rate (without pruned params): {original_total} / {compressed_total} '
               f'({ original_total / compressed_total:.3f} X) \n' 
               f'compression rate (include pruned params): { part_model_original_bytes} / {compressed_total} '
               f'({ part_model_original_bytes / compressed_total:.3f} X)')
    util.log(f"{args.save_dir}/{args.log}", log_str)
    print('-' * 70)
    print(log_str)



