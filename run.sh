#!/usr/bin/env bash
# ----------------------- MESA-fc ------------------------------
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.0 -ep 200 -qep 40 -sd model_mesa_0.0_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -ep 200 -qep 40 -sd model_mesa_0.1_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.01 -ep 200 -qep 40 -sd model_mesa_0.01_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.001 -ep 200 -qep 40 -sd model_mesa_0.001_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.0001 -ep 200 -qep 40 -sd model_mesa_0.0001_fc

# ----------------------- MESA2-conv ------------------------------
# norm-base
python main.py -mm c -prunep -quanp -encp -b conv=8,fc=5 -pm filter-norm -be 0.0 -sd model_mesa_0.0_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar
python main.py -mm c -prunep -quanp -encp -b conv=8,fc=5 -pm filter-norm -be 5.0 -sd model_mesa_5.0_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar
# geometric median
python main.py -mm c -initp -prunep -quanp -encp -b conv=8,fc=5 -pm filter-gm -be 0.0 -sd model_mesa_0.0_filter_gm_conv
python main.py -mm c -initp -prunep -quanp -encp -b conv=8,fc=5 -pm filter-gm -be 1.0 -sd model_mesa_1.0_filter_gm_conv

# ----------------------- all ------------------------------
## train with conv-percentile and fc penalty and prune and quantize
python main.py -mm all -initp -prunep -quanp -encp -pm percentile -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -be 1.0 -sd model_mesa_0.1_beta_1.0_perc_all
#python main.py -mm all -tm 4 -pm filter -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -be 0.1 -lr 0.1 -ep 200 -qep 10 -sd model_mesa_0.1_filter_all -lm model_mesa_0.1_filter_all/checkpoint_initial_alpha_0.1_200.tar
