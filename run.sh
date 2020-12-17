#!/usr/bin/env bash

# ----------------------- DeepC-conv ----------------------------------
python main.py -mm c -prunep -quanp -encp -pm percentile -b conv=5,fc=5 -p fc1=10,fc2=10,fc3=4 -sd model_deepc -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar

# ----------------------- MESA-fc --------------------------------
python main.py -mm d -initp -quanp -encp -umfcm -b conv=5,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.0 -ep 200 -qep 40 -sd model_mesa_0.0_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=5,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -ep 200 -qep 40 -sd model_mesa_0.1_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=5,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.01 -ep 200 -qep 40 -sd model_mesa_0.01_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=5,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.001 -ep 200 -qep 40 -sd model_mesa_0.001_fc
python main.py -mm d -initp -quanp -encp -umfcm -b conv=5,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.0001 -ep 200 -qep 40 -sd model_mesa_0.0001_fc

# ----------------------- MESA2-conv ------------------------------
# ----------- AlexNet ---------------
# norm-base, filter3d-delta loss
python main.py -mm c -prunep -quanp -encp -b conv=5,fc=5 -pm filter-norm -be 0.0 -sd model_mesa_0.0_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar -prlr 0.001 -prep 200
python main.py -mm c -prunep -quanp -encp -b conv=5,fc=5 -pm filter-norm -be 0.1 -sd model_mesa_0.1_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar -prlr 0.001 -prep 200
python main.py -mm c -prunep -quanp -encp -b conv=5,fc=5 -pm filter-norm -be 1.0 -sd model_mesa_1.0_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar -prlr 0.001 -prep 200
python main.py -mm c -prunep -quanp -encp -b conv=5,fc=5 -pm filter-norm -be 2.0 -sd model_mesa_2.0_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar -prlr 0.001 -prep 200
# norm-base, position mean loss
python main.py -mm c -prunep -quanp -encp -b conv=5,fc=5 -pm filter-norm -be 0.1 -clf position-mean -sd model_mesa_posMean_0.1_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar -prlr 0.001 -prep 200
# norm-base, width1d delta loss
python main.py -mm c -prunep -quanp -encp -b conv=5,fc=5 -pm filter-norm -be 0.1 -clf width1d-delta -sd model_mesa_width1dDelta_0.1_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar -prlr 0.001 -prep 200
python main.py -mm c -prunep -quanp -encp -b conv=5,fc=5 -pm filter-norm -be 0.01 -clf width1d-delta -sd model_mesa_width1dDelta_0.01_filter_norm_conv -lm model_mesa_AlexNet/checkpoint_initial_alpha_0.1_149.tar -prlr 0.001 -prep 200
# geometric median, delta loss
python main.py -mm c -initp -prunep -quanp -encp -b conv=5,fc=5 -pm filter-gm -be 0.0 -sd model_mesa_0.0_filter_gm_conv
python main.py -mm c -initp -prunep -quanp -encp -b conv=5,fc=5 -pm filter-gm -be 1.0 -sd model_mesa_1.0_filter_gm_conv

# ----------- VggNet ---------------
# norm-base
python main.py -mm c -prunep -quanp -um vgg16 -pm filter-norm -be 0.1 -b conv=5,fc=5 -sd model_mesa_0.0_vgg_filter_norm

# ----------------------- all ------------------------------
## train with conv-percentile and fc penalty and prune and quantize
python main.py -mm all -initp -prunep -quanp -encp -pm percentile -b conv=5,fc=5 -p fc1=10,fc2=10,fc3=4 -al 0.1 -be 1.0 -sd model_mesa_0.1_beta_1.0_perc_all
