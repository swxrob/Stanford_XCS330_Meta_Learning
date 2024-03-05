#!/bin/bash

ARG_DEVICE=--device=${1:-'cpu'}

ARG_COMPILE=--${2:-'no-compile'}

ARG_BACKEND=--backend=${3:-'inductor'}

ARG_BATCHSIZE=--meta_batch_size=${4:-'128'}

# LSTM Memory Size = 8
python main.py --num_shot 1 --num_classes 3 --hidden_dim 8   $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE

# LSTM Memory Size = 128
python main.py --num_shot 1 --num_classes 3 --hidden_dim 128 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE

# LSTM Memory Size = 256
python main.py --num_shot 1 --num_classes 3 --hidden_dim 256 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE

