import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path
import sys

import torch
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import deep_compression.util

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq

def huffman_encode(arr,save_dir='./'):

    # Infer dtype
    dtype = str(arr.dtype)

    # Calculate frequency in arr
    freq_map = defaultdict(int)
    convert_map = {'float32':float, 'int32':int}
    for value in np.nditer(arr):
        value = convert_map[dtype](value)
        freq_map[value] += 1

    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)

    # Merge nodes
    while(len(heap) > 1):
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
    datasize = dump(data_encoding)
    #print(data_encoding)


    # Dump codebook (huffman tree)
    codebook_encoding = encode_huffman_tree(root, dtype)

    treesize = dump(codebook_encoding)

    return treesize, datasize

# Logics to encode / decode huffman tree
# Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
def encode_huffman_tree(root, dtype):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    converter = {'float32':float2bitstr, 'int32':int2bitstr}
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
def dump(code_str):
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
    header = "{:08b}".format(num_of_padding)
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))
    # Dump to a file
    return len(byte_arr)

def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    with open(filename, 'rb') as f:
        header = f.read(1)
        rest = f.read() # bytes
        code_str = ''.join('{byte:08b}' for byte in rest)
        offset = ord(header)
        if offset != 0:
            code_str = code_str[:-offset] # string of '0's and '1's
    return code_str


# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes

    return ''.join('{:08b}'.format(byte) for byte in four_bytes) # string of '0's and '1's

def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]

def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer) # bytes
    return ''.join('{:08b}'.format(byte) for byte in four_bytes) # string of '0's and '1's

def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr, maxdiff):
    org_diff = indptr[1:] - indptr[:-1]
    diff = []
    for v in org_diff:
        n = v/maxdiff
        for i in range(int(n)):
            diff.append(maxdiff)
            v-=maxdiff
        diff.append(v)
    return np.array(diff)

def calc_indice_diff(indices,maxdiff): 
    org_diff = indices[1:] - indices[:-1]
    diff = []
    for v in org_diff:
        if v>0:
            n = v/maxdiff
            for i in range(int(n)):
                diff.append(maxdiff)
                v-=maxdiff
            diff.append(v)
        else:
            diff.append(abs(v))
    return np.array(diff)
def reconstruct_indptr(diff):
    return np.concatenate([[0], np.cumsum(diff)])


# Encode / Decode models
def huffman_encode_model(model, maxdiff, directory='encodings/'):
    original_total = 0
    compressed_total = 0
    
    weight = model
    shape = weight.shape
    mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

    print('-'*70)
    t0, d0 = huffman_encode(mat.data, directory)
    t1, d1 = huffman_encode(calc_indice_diff(mat.indices,maxdiff).astype(np.float32), directory)
    t2, d2 = huffman_encode(calc_index_diff(mat.indptr, maxdiff).astype(np.float32), directory)
    original = weight.data.nbytes
    compressed = t0 + t1 + t2 + d0 + d1 + d2
       
    original_total = original
    compressed_total = compressed

    print('original:{} bytes;  after:{} bytes'.format(original_total, compressed_total))

    return original_total, compressed_total
def huffman_encode_model_no_sparse_matrix(model, directory='encodings/'):
    original_total = 0
    compressed_total = 0

    print('-'*70)
    model = model.flatten()
    t0, d0 = huffman_encode(model, directory)


    # Print statistics
    original = model.nbytes
    compressed = t0 + d0

    original_total = original
    compressed_total = compressed

    print('original:{} bytes;  after:{} bytes'.format(original_total, compressed_total))

    return original_total, compressed_total

