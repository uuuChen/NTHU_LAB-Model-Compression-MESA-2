import argparse

import torch
import numpy
from numpy import array
from mpd.huffmancoding import huffman_encode_model
import mpd.util as util
import mpd.AlexNet_mask as AlexNet_mask


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


def to_index(layer):
    # print(1 - numpy.sum(numpy.where(layer == 0, True, False).reshape(-1)) / layer.reshape(-1).shape[0])
    set_=numpy.unique(layer)
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
    return numpy.vectorize(dict_.get)(layer)


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


# def matrix_cycledistance(array_a,array_b,size_x,size_y,maxdistance):
#     distancearray=numpy.zeros(shape=(size_x,size_y))
#     for i in range(size_x):
#         for j in range(size_y):
#             distancearray[i][j]=cycledistance(array_a[i][j],array_b[i][j],maxdistance)
#     return distancearray


def matrix_cycledistance(array_a, array_b, maxdistance):
    flat_list_a = list(array_a.reshape(-1))
    flat_list_b = list(array_b.reshape(-1))
    distancearray = list()
    for a_val, b_val in list(zip(flat_list_a, flat_list_b)):
        distancearray.append(cycledistance(a_val, b_val, maxdistance))
    distancearray = numpy.array(distancearray).reshape(array_a.shape)
    return distancearray


def conv_editgraph_and_firstfilter(conv, maxdistance):
    pruned_filter_indice = numpy.where(numpy.sum(conv.reshape(conv.shape[0], -1), axis=1) == 0)[0]
    left_filter_indice = numpy.array(list(set(range(conv.shape[0])).difference(pruned_filter_indice)))
    left_filters = conv[left_filter_indice, :, :, :]
    first_filter = None
    prev_filter = None
    edit_histogram = []
    for cur_filter_idx, cur_filter in list(enumerate(left_filters)):
        if cur_filter_idx == 0:
            first_filter = cur_filter
        else:
            edit_histogram.append(matrix_cycledistance(prev_filter, cur_filter, maxdistance))
        prev_filter = cur_filter
    edit_histogram = numpy.array(edit_histogram)
    return edit_histogram, first_filter, pruned_filter_indice


def getblocks(fc,x_axis, y_axis, partitionsize, maxdistance):
    block_histogram=[]
    for i in range(partitionsize):
        blocks=[]
        for j in range(x_axis//partitionsize):
            for k in range(y_axis//partitionsize):
                blocks.append(fc[j][k])
        if i==0:
            block_histogram = array(blocks)
        else:
            block_histogram = numpy.concatenate((block_histogram,array(blocks)), axis=0)
    return block_histogram


def editgraph_and_firstblock(fc,x_axis, y_axis, partitionsize, maxdistance):
    edit_histogram=[]
    for i in range(partitionsize-1):
        if i == 0:
            edit_histogram = matrix_cycledistance(
                fc[
                    i * (x_axis // partitionsize):(i + 1) * (x_axis // partitionsize):1,
                    i * (y_axis // partitionsize):(i + 1) * (y_axis // partitionsize):1
                ],
                fc[
                    (i + 1) * (x_axis // partitionsize):(i + 2) * (x_axis // partitionsize):1,
                    (i + 1) * (y_axis // partitionsize):(i + 2) * (y_axis // partitionsize):1
                ],
                maxdistance
            )
        else:
            temp = matrix_cycledistance(
                fc[
                    i * (x_axis // partitionsize):(i + 1) * (x_axis // partitionsize):1,
                    i * (y_axis // partitionsize):(i + 1) * (y_axis // partitionsize):1
                ],
                fc[
                    (i + 1) * (x_axis // partitionsize):(i + 2) * (x_axis // partitionsize):1,
                    (i + 1) * (y_axis // partitionsize):(i + 2) * (y_axis // partitionsize):1
                ],
                maxdistance
            )
            edit_histogram=numpy.concatenate((edit_histogram, temp), axis=0)
    # for first block
    first_block=[]
    for j in range(x_axis//partitionsize):
        for k in range(y_axis//partitionsize):
            first_block.append(fc[j][k])
    return edit_histogram, array(first_block)


def mpd_huffman_encode(val_loader ,model_path ,args):
    # alpha = float(model_path.split("_")[4])
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    org_total = 0
    first_total = 0
    edit_total = 0
    indice_total = 0
    first_compressed = 0
    edit_compressed = 0
    indice_compressed = 0
    fc_t =0
    fc_d =0
    size = 0

    fc_compressed_without_edit = 0

    fc_org_total = 0
    fc_compressed = 0
    conv_org_total = 0
    conv_compressed = 0
    real_fc_org = 0
    edit_distance_fc_list = []
    layer_compressed_dic={}
    layer_org_dic={}

    if args.model_mode == 'c':
        model = AlexNet_mask.AlexNet(mask_flag=True).cuda()
    else:
        model = AlexNet_mask.AlexNet_mask('AlexNet_mask', args.partition, mask_flag=True).cuda()
    model = util.load_checkpoint(model,  f"{model_path}",args)
    ori_model_bytes = get_ori_model_bytes(model, args)

    util.log(f"{args.log_detail}", f"\n\n-----------------------------------------")
    util.log(f"{args.log_detail}", f"{model_path}")
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'bn' in name:
            continue
        if 'bias' in name:
            continue
        if 'conv' in name and args.model_mode is not 'd':
            print(f'\n{name}')
            weight = param.data.cpu().numpy()
            conv_index = to_index(weight)
            size += conv_index.size
            conv_edit, conv_first_filter, pruned_filter_indice = conv_editgraph_and_firstfilter(conv_index, 2**int(args.bits['conv']))
            print('---------first filter-----------')
            conv_byte, conv_compressed_byte, t0, d0=huffman_encode_model(conv_first_filter.astype(numpy.int32))
            first_total += conv_byte
            conv_org_total += conv_byte
            first_compressed += conv_compressed_byte
            conv_compressed += conv_compressed_byte
            util.log(f"{args.log_detail}", f"{name}")
            util.log(f"{args.log_detail}", f"\tfirst original:{conv_byte} bytes\t fist filter after:{conv_compressed_byte} bytes")
            print('---------edit filter-----------')
            conv_edit_bytes, conv_edit_compress, t0, d0=huffman_encode_model(conv_edit.astype(numpy.float32))
            edit_total += conv_edit_bytes
            conv_org_total += conv_edit_bytes
            edit_compressed += conv_edit_compress
            conv_compressed += conv_edit_compress
            util.log(f"{args.log_detail}", f"\tedit original:{conv_edit_bytes} bytes\t edit filter after:{conv_edit_compress} bytes")
            util.log(f"{args.log_detail}", f"original:{conv_org_total} bytes\t after:{conv_compressed} bytes")
            print('---------pruned filter indice-----------')
            conv_indice_bytes, conv_indice_compress, t0, d0 = huffman_encode_model(pruned_filter_indice.astype(numpy.float32))
            indice_total += conv_indice_bytes
            conv_org_total += conv_indice_bytes
            indice_compressed += conv_indice_compress
            conv_compressed += conv_indice_compress
            util.log(f"{args.log_detail}",
                     f"\tindice original:{conv_edit_bytes} bytes\t indice after:{conv_edit_compress} bytes")
            util.log(f"{args.log_detail}", f"original:{conv_org_total} bytes\t after:{conv_compressed} bytes")
        if 'fc' in name and args.model_mode is not 'c':
            print(name)
            partition = int(args.partition[name[0:3]])
            weight = param.data.cpu().numpy()
            fc_index = to_index(weight)
            # ----------- compress with editgraph format ------------------------------------------
            size += fc_index.size
            fc_edit, fc_first_block = editgraph_and_firstblock(fc_index,numpy.size(fc_index,0),numpy.size(fc_index,1),partition,2**int(args.bits['fc']))
            print('---------first block-----------')
            fc_bytes, fc_compressed_bytes, t0, d0=huffman_encode_model(fc_first_block.astype(numpy.int32))
            fc_t += t0
            fc_d += d0
            first_total += fc_bytes
            fc_org_total += fc_bytes
            first_compressed += fc_compressed_bytes
            fc_compressed += fc_compressed_bytes
            util.log(f"{args.log_detail}", f"{name}")
            util.log(f"{args.log_detail}", f"\tfirst original:{fc_bytes} bytes\t fist block after:{fc_compressed_bytes} bytes")
            print('---------edit block-----------')
            fc_edit_bytes, fc_edit_compress, t0, d0=huffman_encode_model(fc_edit.astype(numpy.float32))
            fc_t += t0
            fc_d += d0
            edit_total += fc_edit_bytes
            fc_org_total += fc_edit_bytes
            edit_compressed += fc_edit_compress
            fc_compressed += fc_edit_compress
            util.log(f"{args.log_detail}", f"\tedit original:{fc_edit_bytes} bytes\t edit block after:{fc_edit_compress} bytes")
            util.log(f"{args.log_detail}", f"original:{fc_bytes+fc_edit_bytes} bytes\t after:{fc_compressed_bytes+fc_edit_compress} bytes")
            layer_compressed_dic[name[0:3]] = fc_compressed_bytes + fc_edit_compress

            # ----------- compress without editgraph format ------------------------------------------
            size += fc_index.size
            blocks = getblocks(fc_index,numpy.size(fc_index,0),numpy.size(fc_index,1),partition,2**int(args.bits['fc']))

            fc_bytes, fc_compressed_bytes, t0, d0=huffman_encode_model(blocks.astype(numpy.int32))
            fc_compressed_without_edit += fc_compressed_bytes
            layer_org_dic[name[0:3]] = fc_bytes

            # ----------- count edit distance ------------------------------------------
            shape = weight.shape
            edit_distance=0
            for i in range(partition-1):
                for j in range(shape[0]//partition):
                    for k in range(shape[1]//partition):
                        if(numpy.absolute(weight[i*(shape[0]//partition)+j][i*(shape[1]//partition)+k]-weight[(i+1)*(shape[0]//partition)+j][(i+1)*(shape[1]//partition)+k])!=0):
                            edit_distance+=1
            edit_distance_fc_list.append(edit_distance)

    print('\nfirst_block bytes org:{}'.format(first_total))
    print('first_block bytes after compression:{}'.format(first_compressed))
    print('edit_total bytes org:{}'.format(edit_total))
    print('edit_total bytes after compression:{}\n'.format(edit_compressed))

    compressed = conv_compressed + fc_compressed
    if args.model_mode == 'c' and ('filter' in args.prune_mode or 'channel' in args.prune_mode):
        compressed += indice_compressed

    util.log(f"{args.log}", f"\n\n------------------------------------")
    util.log(f"{args.log}", f"{model_path}")

    util.log(f"{args.log}", f"\noriginal total:{ori_model_bytes} bytes")
    util.log(f"{args.log}", f"compress total:{compressed} bytes")
    util.log(f"{args.log}", f"compressed rate:{(ori_model_bytes/compressed)}")
    if args.model_mode != 'c':
        util.log(f"{args.log}", f"fc original total: {fc_org_total} bytes")
        util.log(f"{args.log}", f"fc compress total: {fc_compressed} bytes")
    if args.model_mode != 'd':
        util.log(f"{args.log}", f"conv original total: {conv_org_total} bytes")
        util.log(f"{args.log}", f"conv compress total: {conv_compressed} bytes")
    util.log(f"{args.log}", f"average bits:{compressed*8/(size)} bits")

    print('original bytes:{}'.format(ori_model_bytes))
    print('bytes after compression:{}'.format(compressed))
    print('compressed rate:{}'.format(ori_model_bytes/compressed))
    print('average bit:{}'.format(compressed*8/size))

    acc = None
    return acc, ori_model_bytes,  compressed, fc_compressed_without_edit, edit_distance_fc_list, fc_t, fc_d, layer_compressed_dic, layer_org_dic
