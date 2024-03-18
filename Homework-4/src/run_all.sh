#!/bin/bash

ARG_DEVICE=--device=${1:-'cpu'}

#####
# Q1
#####
python main.py --task run_ft --model bert-tiny,bert-med --dataset amazon --k 1,8,128 $ARG_DEVICE
python main.py --task plot_ft --model bert-tiny,bert-med --dataset amazon --k 1,8,128 $ARG_DEVICE
mv plot.png plot_q1-ft.png

#####
# Q2 
#####
python main.py --task run_icl --model med,full --dataset babi --k 0,1,16 $ARG_DEVICE
python main.py --task plot_icl --model med,full --dataset babi --k 0,1,16 $ARG_DEVICE
mv plot.png plot_q2-icl-babi.png

# Takes around 8h on the provided VM with a Tesla K80 GPU
python main.py --task run_icl --model med,full --dataset xsum --k 0,1,4 --prompt none,tldr,custom $ARG_DEVICE
python main.py --task plot_icl --model med,full --dataset xsum --k 0,1,4 --prompt none,tldr,custom $ARG_DEVICE
mv plot.png plot_q2-icl-xsum.png

#####
# Q3: takes around 3h on the provided VM with a Tesla K80 GPU
#####
# Fine-Tune Train on XSum and Babi for several model layers and K-shots
python main.py --task run_ft --model med --mode first,last,middle,lora4,lora16 --dataset xsum,babi --k 0,1,8,128 $ARG_DEVICE

# Plot for XSum
python main.py --task plot_ft --model med --mode first,last,middle,lora4,lora16 --dataset xsum --k 0,1,8,128 $ARG_DEVICE
mv plot.png plot_q3-ft-xsum.png

# Plot for Babi
python main.py --task plot_ft --model med --mode first,last,middle,lora4,lora16 --dataset babi --k 0,1,8,128 $ARG_DEVICE
mv plot.png plot_q3-ft-babi.png

#####
# Q4
#####
# LoRA-16 versus In-Context Learning on XSum
python main.py --task plot $ARG_DEVICE
mv plot.png plot_q4-lora16-vs-icl-xsum.png

# Ordering test: Stats
python main.py --task run_icl --model med --dataset babi --k 16 --repeats 5 $ARG_DEVICE

