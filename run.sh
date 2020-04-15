#!/usr/bin/env bash
# ----------------------- MESA ------------------------------
python main.py -mm d -initp -quanp -fc-mask -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.0 -ep 200 -qep 40 -sd model_mesa_0.0_fc
python main.py -mm d -initp -quanp -fc-mask -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -ep 200 -qep 40 -sd model_mesa_0.1_fc
python main.py -mm d -initp -quanp -fc-mask -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.01 -ep 200 -qep 40 -sd model_mesa_0.01_fc
python main.py -mm d -initp -quanp -fc-mask -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.001 -ep 200 -qep 40 -sd model_mesa_0.001_fc
python main.py -mm d -initp -quanp -fc-mask -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.0001 -ep 200 -qep 40 -sd model_mesa_0.0001_fc

# ----------------------- conv ------------------------------
# train with conv penalty
python main.py -mm c -prunep -quanp -b 5 -pm filter-norm -be 1.0 -sd model_mesa_1.0_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar
# train with conv geometric-median
python main.py -mm c -initp -prunep -quanp -b 5 -pm filter-gm -be 1.0 -sd model_mesa_1.0_filter_gm_conv

# ----------------------- fc ------------------------------
python main.py -mm d -tm 5 -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.1  -lr 0.1  -ep 200 -qep 10 -sd model_mesa_0.1_fc

# ----------------------- all ------------------------------
# train with conv-percentile and fc penalty and prune and quantize
python main.py -mm all -tm 4 -pm percentile -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -be 0.1 -lr 0.1 -ep 200 -qep 10 -sd model_mesa_0.1_perc_all -lm model_mesa_0.1_perc_all/checkpoint_initial_alpha_0.1_200.tar
python main.py -mm all -tm 4 -pm filter -b 5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -be 0.1 -lr 0.1 -ep 200 -qep 10 -sd model_mesa_0.1_filter_all -lm model_mesa_0.1_filter_all/checkpoint_initial_alpha_0.1_200.tar

# ----------------------- compute compressed rate ------------------------------
python main.py -mm d -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -lmm checkpoint_quantized_re_alpha_0.1_20.tar
python main.py -mm all -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -lmm checkpoint_quantized_re_alpha_0.1_20.tar
python main.py -mm c -b conv=8,fc=5 -pm filter-norm -p fc1=10,fc2=10,fc3=4 -lmm checkpoint_quantized_re_beta_1.0_19.tar
python main.py -mm c -b conv=8,fc=5 -pm filter-gm -p fc1=10,fc2=10,fc3=4 -lmm checkpoint_quantized_alpha_0.1_188.tar