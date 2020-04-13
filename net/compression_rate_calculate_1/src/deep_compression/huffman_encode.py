import argparse

import torch
import torch.nn as nn
import numpy
from numpy import array
from deep_compression.huffmancoding import huffman_encode_model, huffman_encode_model_no_sparse_matrix
from net.models import AlexNet
import deep_compression.util as util
from net import models

def get_ori_model_bytes(model, args):
    model_params = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'bn' in name:
            continue
        if 'bias' in name:
            continue
        if 'conv' in name and args.model_mode is not 'd':
            model_params += param.reshape(-1).shape[0]
        if 'fc' in name and args.model_mode is not 'c':
            model_params += param.reshape(-1).shape[0]
    model_bytes = model_params * 4
    return model_bytes

def to_index(fc):
    set_=numpy.unique(fc)
    dict_={}
    count_neg=0
    for i in range(len(set_)):
        if set_[i] <0:
            dict_[set_[i]] = len(set_)-1- i
            count_neg+=1
        elif set_[i] == 0:
            dict_[set_[i]] = 0
        else:
            dict_[set_[i]]=i-count_neg
    return numpy.vectorize(dict_.get)(fc)

def cycledistance(a,b,maxdis):
    distance=abs(a-b)
    if(a>b): #4->2
        if(distance<abs(maxdis-distance)):
            return (-1)*distance
        else:
            return abs(maxdis-distance)
    elif(a<b): #2->4
        if(distance<abs(maxdis-distance)):
            return distance
        else:
            return (-1)*abs(maxdis-distance)
    else:
        return 0

def matrix_cycledistance(array_a,array_b,size_x,size_y,maxdistance):
    distancearray=numpy.zeros(shape=(size_x,size_y))
    for i in range(size_x):
        for j in range(size_y):
            distancearray[i][j]=cycledistance(array_a[i][j],array_b[i][j],maxdistance)
    return distancearray

def editgraph_and_firstblock(fc,x_axis, y_axis, partitionsize, maxdistance):
    edit_histogram=[]
    for i in range(partitionsize-1):
        if i==0:
            edit_histogram=matrix_cycledistance(fc[i*(x_axis//partitionsize):(i+1)*(x_axis//partitionsize):1,i*(y_axis//partitionsize):(i+1)*(y_axis//partitionsize):1],
            fc[(i+1)*(x_axis//partitionsize):(i+2)*(x_axis//partitionsize):1,(i+1)*(y_axis//partitionsize):(i+2)*(y_axis//partitionsize):1],
            x_axis//partitionsize,
            y_axis//partitionsize,
            maxdistance)

        else:
            temp=matrix_cycledistance(fc[i*(x_axis//partitionsize):(i+1)*(x_axis//partitionsize):1,i*(y_axis//partitionsize):(i+1)*(y_axis//partitionsize):1],
            fc[(i+1)*(x_axis//partitionsize):(i+2)*(x_axis//partitionsize):1,(i+1)*(y_axis//partitionsize):(i+2)*(y_axis//partitionsize):1],
            x_axis//partitionsize,
            y_axis//partitionsize,
            maxdistance)
            edit_histogram=numpy.concatenate((edit_histogram,temp),axis=0)

    first_block=[]
    for j in range(x_axis//partitionsize):
        for k in range(y_axis//partitionsize):
            first_block.append(fc[j][k])

    return edit_histogram, array(first_block)


def deep_huffman_encode(val_loader, model_path, sparse_mode, compress_flag,args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    maxdistance=2**args.index_bits
    org_total = 0
    compressed = 0
    fc_org_total = 0
    fc_compressed_total = 0
    conv_org_total = 0
    conv_compressed_total = 0
    size = 0
    # model = torch.load(model_path)
    model = AlexNet(mask_flag=True).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    ori_model_bytes = get_ori_model_bytes(model, args)
    criterion = nn.CrossEntropyLoss().cuda()
    acc = util.validate(args, val_loader, model, criterion)
    if compress_flag == False:
        return acc, fc_org_total,  fc_compressed_total
    util.log(f"{args.log_detail}", f"\n\n--------------------------------")
    util.log(f"{args.log_detail}", f"{model_path}\t sprase mode:{sparse_mode}")
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'bn' in name:
            continue
        if 'bias' in name:
            continue
        if 'conv' in name and args.model_mode is not 'd':
            print(name)
            weight = param.data.cpu().numpy()
            weight = weight.reshape(weight.shape[0]*weight.shape[1],-1)
            size += weight.size
            if sparse_mode == 't':
                fc_bit, fc_compressed_bit = huffman_encode_model(weight.astype(numpy.float32), maxdistance)
            else:
                fc_bit, fc_compressed_bit = huffman_encode_model_no_sparse_matrix(weight.astype(numpy.float32))
            util.log(f"{args.log_detail}", f"{name}")
            util.log(f"{args.log_detail}", f"original:{fc_bit} bytes\t after:{fc_compressed_bit} bytes")
            org_total += fc_bit
            compressed += fc_compressed_bit
            conv_org_total += fc_bit
            conv_compressed_total += fc_compressed_bit
        if 'fc' in name and args.model_mode is not 'c':
            print(name)
            weight = param.data.cpu().numpy()
            weight = to_index(weight)
            size += weight.size
            if sparse_mode == 't':
                fc_bit, fc_compressed_bit = huffman_encode_model(weight.astype(numpy.float32), maxdistance)
            else:
                fc_bit, fc_compressed_bit = huffman_encode_model_no_sparse_matrix(weight.astype(numpy.float32))
            util.log(f"{args.log_detail}", f"{name}")
            util.log(f"{args.log_detail}", f"original:{fc_bit} bytes\t after:{fc_compressed_bit} bytes")
            org_total += fc_bit
            compressed += fc_compressed_bit
            fc_org_total += fc_bit
            fc_compressed_total += fc_compressed_bit
    util.log(f"{args.log}", f"\n\n--------------------------------")
    util.log(f"{args.log}", f"{model_path}\t sprase mode:{sparse_mode}")
    util.log(f"{args.log}", f"fc original total: {fc_org_total} bytes")
    util.log(f"{args.log}", f"fc compress total: {fc_compressed_total} bytes")
    util.log(f"{args.log}", f"conv original total: {conv_org_total} bytes")
    util.log(f"{args.log}", f"conv compress total: {conv_compressed_total} bytes")
    util.log(f"{args.log}", f"original total:{org_total} bytes")
    util.log(f"{args.log}", f"compress total:{compressed} bytes")
    util.log(f"{args.log}", f"compressed rate:{(org_total/compressed)}")
    util.log(f"{args.log}", f"average bit:{compressed*8/(size)} bits")
    print('original bytes:{}'.format(org_total))
    print('bytes after compression:{}'.format(compressed))
    print('compressed rate:{}'.format(org_total/compressed))
    print('average bit:{}'.format(compressed*8/(size)))

    return acc, ori_model_bytes,  compressed
