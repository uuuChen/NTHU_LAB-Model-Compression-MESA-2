import os
import torch
import numpy as np
import struct
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
from pathlib import Path

from scipy.sparse import csr_matrix, csc_matrix

import util

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq


def huffman_encode(arr, prefix, save_dir='./'):
    """
    Encodes numpy array 'arr' and saves to `save_dir`
    The names of binary files are prefixed with `prefix`
    returns the number of bytes for the tree and the data after the compression
    """
    # Infer dtype
    dtype = str(arr.dtype)

    # Calculate frequency in arr
    freq_map = defaultdict(int)
    convert_map = {'float32': float, 'int32': int}
    for value in np.nditer(arr):
        value = convert_map[dtype](value)
        freq_map[value] += 1

    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)

    # Merge nodes
    while len(heap) > 1:
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.freq + node2.freq, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = {}

    def generate_code(node, code):
        if node is None:
            return
        if node.value is not None:
            value2code[node.value] = code
            return
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')

    root = heappop(heap)
    generate_code(root, '')

    # Path to save location
    directory = Path(save_dir)

    # Dump data
    data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
    datasize = dump(data_encoding, directory/f'{prefix}.bin')

    # Dump codebook (huffman tree)
    codebook_encoding = encode_huffman_tree(root, dtype)
    treesize = dump(codebook_encoding, directory/f'{prefix}_codebook.bin')

    return treesize, datasize


def huffman_decode(directory, prefix, dtype):
    """
    Decodes binary files from directory
    """
    directory = Path(directory)

    # Read the codebook
    codebook_encoding = load(directory/f'{prefix}_codebook.bin')
    root = decode_huffman_tree(codebook_encoding, dtype)

    # Read the data
    data_encoding = load(directory/f'{prefix}.bin')

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value is not None: # Leaf node
            data.append(ptr.value)
            ptr = root

    return np.array(data, dtype=dtype)


# Logics to encode / decode huffman tree
# Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
def encode_huffman_tree(root, dtype):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    converter = {'float32': float2bitstr, 'int32': int2bitstr}
    code_list = []

    def encode_node(node):
        if node.value is not None: # node is leaf node
            code_list.append('1')
            lst = list(converter[dtype](node.value))
            code_list.extend(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)

    encode_node(root)
    return ''.join(code_list)


def decode_huffman_tree(code_str, dtype):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    converter = {'float32':bitstr2float, 'int32':bitstr2int}
    idx = 0

    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1': # Leaf node
            value = converter[dtype](code_str[idx:idx+32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()


# My own dump / load logics
def dump(code_str, filename):
    """
    code_str : string of either '0' and '1' characters
    this function dumps to a file
    returns how many bytes are written
    """
    # Make header (1 byte) and add padding to the end
    # Files need to be byte aligned.
    # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
    # This introduces minimum of 8 bits, maximum of 15 bits overhead
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    with open(filename, 'wb') as f:
        f.write(byte_arr)
    return len(byte_arr)


def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    with open(filename, 'rb') as f:
        header = f.read(1)
        rest = f.read() # bytes
        code_str = ''.join(f'{byte:08b}' for byte in rest)
        offset = ord(header)
        if offset != 0:
            code_str = code_str[:-offset] # string of '0's and '1's
    return code_str


# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's


def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]


def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer)  # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's


def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr):
    return indptr[1:] - indptr[:-1]


def reconstruct_indptr(diff):
    return np.concatenate([[0], np.cumsum(diff)])


def huffman_encode_conv4d(weight, name, directory):  # add
    indices_arr = indptr_arr = len_arr = np.array([], dtype='int32')
    data_arr = np.array([], dtype='float32')
    form = 'csr' if weight.shape[2] < weight.shape[3] else 'csc'
    for k in range(weight.shape[0]):
        for c in range(weight.shape[1]):
            weight2d = weight[k, c, :, :]
            mat = csr_matrix(weight2d) if form == 'csr' else csc_matrix(weight2d)
            data_arr = np.append(data_arr, mat.data)
            indices_arr = np.append(indices_arr, mat.indices)
            indptr_arr = np.append(indptr_arr, calc_index_diff(mat.indptr))
            len_arr = np.append(len_arr, mat.data.shape[0])

    # Encode
    t0, d0 = huffman_encode(data_arr, name + f'_{form}_data', directory)
    t1, d1 = huffman_encode(indices_arr, name + f'_{form}_indices', directory)
    t2, d2 = huffman_encode(indptr_arr, name + f'_{form}_indptr', directory)
    t3, d3 = huffman_encode(len_arr, name + f'_{form}_length', directory)

    # Print statistics
    original = data_arr.nbytes + indices_arr.nbytes + indptr_arr.nbytes + len_arr.nbytes
    compressed = t0 + t1 + t2 + t3 + d0 + d1 + d2 + d3

    return original, compressed


def huffman_decode_conv4d(shape, name, directory):  # add
    form = 'csr' if shape[2] < shape[3] else 'csc'
    matrix = csr_matrix if shape[2] < shape[3] else csc_matrix

    # Decode data
    data_arr = huffman_decode(directory, name + f'_{form}_data', dtype='float32')
    indices_arr = huffman_decode(directory, name + f'_{form}_indices', dtype='int32')
    indptr_arr = huffman_decode(directory, name + f'_{form}_indptr', dtype='int32')
    len_arr = huffman_decode(directory, name + f'_{form}_length', dtype='int32')

    # Construct matrix
    weight = np.zeros(shape=shape)
    len_acc_arr = reconstruct_indptr(len_arr)
    count = 0
    for k in range(shape[0]):
        for c in range(shape[1]):
            start, end = len_acc_arr[count], len_acc_arr[count + 1]
            data = data_arr[start: end]
            indices = indices_arr[start: end]
            indptr = reconstruct_indptr(indptr_arr[count * shape[2]: (count + 1) * shape[3]])
            weight[k, c, :, :] = matrix((data, indices, indptr), (shape[2], shape[3])).toarray()
            count += 1

    return weight


def huffman_encode_sparse2d(weight, name, directory):
    if np.count_nonzero(weight) == 0:  # all the items in weight are zero
        return None, None

    shape = weight.shape
    form = 'csr' if shape[0] < shape[1] else 'csc'
    mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

    # Encode
    t0, d0 = huffman_encode(mat.data, name + f'_{form}_data', directory)
    t1, d1 = huffman_encode(mat.indices, name + f'_{form}_indices', directory)
    t2, d2 = huffman_encode(calc_index_diff(mat.indptr), name + f'_{form}_indptr', directory)

    # Print statistics
    original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
    compressed = t0 + t1 + t2 + d0 + d1 + d2

    return original, compressed


def huffman_decode_sparse2d(shape, name, directory):
    form = 'csr' if shape[0] < shape[1] else 'csc'
    matrix = csr_matrix if shape[0] < shape[1] else csc_matrix

    # Decode data
    data = huffman_decode(directory, name + f'_{form}_data', dtype='float32')
    indices = huffman_decode(directory, name + f'_{form}_indices', dtype='int32')
    indptr = reconstruct_indptr(huffman_decode(directory, name + f'_{form}_indptr', dtype='int32'))

    # Construct matrix
    weight = matrix((data, indices, indptr), shape).toarray()

    return weight


# Encode / Decode models
def deepc_huffman_encode_model(model, args, directory='encodings/'):
    os.makedirs(directory, exist_ok=True)
    original_total = compressed_total = 0
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
                original, compressed = huffman_encode_conv4d(weight, name, directory)
            elif 'fc' in name and args.model_mode != 'c':
                original, compressed = huffman_encode_sparse2d(weight, name, directory)
            else:
                ignore = True
        else:  # bias
            if ('conv' in name and args.model_mode != 'd' or
                    'fc' in name and args.model_mode != 'c'):
                bias = param.data.cpu().numpy()
                bias.dump(f'{directory}/{name}')
                original = bias.nbytes
                compressed = original
            else:
                ignore = True

        if ignore:
            log_str = f"{name:<15} | {'* pass':>10}"
            print(log_str)
            util.log(f"{args.save_dir}/{args.log}", log_str)
        else:
            print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x "
                  f"{100 * compressed / original:>6.2f}%")
            original_total += original
            compressed_total += compressed

    # Print and log statistics
    part_model_original_bytes = util.get_part_model_original_bytes(model, args)
    log_str = (f'\ncompression rate (without pruned params): {original_total} / {compressed_total} '
               f'({original_total / compressed_total:.3f} X) \n'
               f'compression rate (include pruned params): {part_model_original_bytes} / {compressed_total} '
               f'({part_model_original_bytes / compressed_total:.3f} X)')
    print('-' * 70)
    print(log_str)


def deepc_huffman_decode_model(model, directory='encodings/'):
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'weight' in name:
            dev = param.device
            weight = None
            weight_shape = param.data.cpu().numpy().shape
            if 'conv' in name:
                weight = huffman_decode_conv4d(weight_shape, name, directory)
            elif 'fc' in name:
                weight = huffman_decode_sparse2d(weight_shape, name, directory)
            param.data = torch.from_numpy(weight).to(dev, dtype=torch.float)  # Insert to model
        else:
            dev = param.device
            bias = np.load(directory+'/'+name, allow_pickle=True)
            param.data = torch.from_numpy(bias).to(dev)
    return model
