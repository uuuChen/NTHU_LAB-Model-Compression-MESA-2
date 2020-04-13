#!/usr/bin/env bash
python main.py -mm d -tm 5 -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.0    -lr 0.1  -ep 300 -qep 10 -sd model_mesa_0.0
python main.py -mm d -tm 5 -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.0001 -lr 0.1  -ep 300 -qep 10 -sd model_mesa_0.0001
python main.py -mm d -tm 5 -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.001  -lr 0.1  -ep 300 -qep 10 -sd model_mesa_0.001 
python main.py -mm d -tm 5 -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.01   -lr 0.1  -ep 300 -qep 10 -sd model_mesa_0.01  
python main.py -mm d -tm 5 -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.1    -lr 0.1  -ep 300 -qep 10 -sd model_mesa_0.1


#######################
# vanilla AlexNet
#######################
# train with origin AlexNet
python main.py -mm None -tm 8 -b 5 -al 0.1 -lr 0.1  -ep 150 -qep 10 -sd model_mesa_0.1_vanilla

#######################
# conv
#######################
# train with conv penalty
python main.py -mm c -tm 8 -b 5 -be 0.1 -lr 0.1 -ep 200 -qep 20 -sd model_mesa_0.1_filter_conv
# train from scratch or load checkpoint with conv penalty and prune and quantize
python main.py -mm c -tm 9 -b 5 -pm filter-norm -be 0.1 -lr 0.1 -ep 200 -reep 40 -qep 20 -sd model_mesa_0.1_filter_conv -lm model_mesa_0.1_filter_conv/checkpoint_initial_alpha_0.1_200.tar
# train with conv geometric-median
python main.py -mm c -tm 11 -b 5 -pm filter-gm -be 0.1 -lr 0.1 -ep 200 -reep 40 -qep 20 -sd model_mesa_0.1_filter_gm_conv
python main.py -mm c -tm 9 -b 5 -pm filter-gm -be 1.0 -lr 0.1 -ep 200 -reep 40 -qep 20 -sd model_mesa_1.0_filter_gm_conv -lm model_mesa_0.1_filter_gm_conv/checkpoint_initial_alpha_0.1_188.tar

#######################
# fc
#######################
python main.py -mm d -tm 5 -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.1  -lr 0.1  -ep 200 -qep 10 -sd model_mesa_0.1_fc

#######################
# all
#######################
# train with conv-percentile and fc penalty and prune and quantize
python main.py -mm all -tm 4 -pm percentile -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -be 0.1 -lr 0.1 -ep 200 -qep 10 -sd model_mesa_0.1_perc_all -lm model_mesa_0.1_perc_all/checkpoint_initial_alpha_0.1_200.tar
python main.py -mm all -tm 4 -pm filter -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -be 0.1 -lr 0.1 -ep 200 -qep 10 -sd model_mesa_0.1_filter_all -lm model_mesa_0.1_filter_all/checkpoint_initial_alpha_0.1_200.tar

#######################
# compute compression rate
#######################
python main.py -mm d -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -lmm checkpoint_quantized_re_alpha_0.1_20.tar

python main.py -mm all -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -lmm checkpoint_quantized_re_alpha_0.1_20.tar

python main.py -mm c -b conv=8,fc=5 -pm filter -p fc1=10,fc2=10,fc3=4 -lmm checkpoint_quantized_filter_re_alpha_0.1_19.tar
python main.py -mm c -b conv=8,fc=5 -pm filter-gm -p fc1=10,fc2=10,fc3=4 -lmm checkpoint_quantized_alpha_0.1_188.tar