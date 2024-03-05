#!/bin/bash

ARG_DEVICE=--device=${1:-'cpu'}

ARG_COMPILE=--${2:-'no-compile'}

ARG_BACKEND=--backend=${3:-'inductor'}

ARG_BATCHSIZE=--meta_batch_size=${4:-'128'}

# Different Learning Rates

# Slower
python main.py --num_shot 1 --num_classes 2 --learning_rate 0.0001  $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE
echo "Done with Learning Rate 0.0001, take a screenshot!"
read junk

# Default
python main.py --num_shot 1 --num_classes 2 --learning_rate 0.001 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE
echo "Done with Learning Rate 0.001, take a screenshot!"
read junk

# Faster
python main.py --num_shot 1 --num_classes 2 --learning_rate 0.01   $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE
echo "Done with Learning Rate 0.01, take a screenshot!"
read junk

