#!/usr/bin/env bash

# ----------------------- DeepC-conv ----------------------------------
python main.py -mm c -prunep -quanp -encp -pm percentile -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -sd model_deepc -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar

# ----------------------- MESA-fc --------------------------------
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.0 -ep 200 -qep 40 -sd model_mesa_0.0_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -ep 200 -qep 40 -sd model_mesa_0.1_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.01 -ep 200 -qep 40 -sd model_mesa_0.01_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.001 -ep 200 -qep 40 -sd model_mesa_0.001_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.0001 -ep 200 -qep 40 -sd model_mesa_0.0001_fc

# ----------------------- MESA2-conv ------------------------------
# ----------- AlexNet ---------------
# norm-base
python main.py -mm c -prunep -quanp -encp -b conv=8,fc=5 -pm filter-norm -be 0.0 -sd model_mesa_0.0_filter_norm_conv
python main.py -mm c -prunep -quanp -encp -b conv=8,fc=5 -pm filter-norm -be 5.0 -sd model_mesa_5.0_filter_norm_conv
# geometric median
python main.py -mm c -initp -prunep -quanp -encp -b conv=8,fc=5 -pm filter-gm -be 0.0 -sd model_mesa_0.0_filter_gm_conv
python main.py -mm c -initp -prunep -quanp -encp -b conv=8,fc=5 -pm filter-gm -be 1.0 -sd model_mesa_1.0_filter_gm_conv

# ----------- VggNet ---------------
# norm-base
python main.py -mm c -prunep -quanp -um vgg16 -pm filter-norm -be 0.1 -b conv=8,fc=5 -sd model_mesa_0.0_vgg_filter_norm

# ----------------------- all ------------------------------
## train with conv-percentile and fc penalty and prune and quantize
python main.py -mm all -initp -prunep -quanp -encp -pm percentile -b conv=8,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -be 1.0 -sd model_mesa_0.1_beta_1.0_perc_all
