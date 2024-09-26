#!/bin/sh
dataset_name=$1
model_name=$2
repo=$3
dropout=$4
temperature=$5
port=$6

torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:${port} \
run.py \
--model_name_or_path ${repo}/${model_name} \
--output_dir results \
--train_data ${model_name}/${dataset_name}${dropout}.json \
--same_task_within_batch True \
--sentence_pooling_method mean \
--do_train False \
--temperature ${temperature} \
> ${model_name}/${dataset_name}-result${dropout}-temp-${temperature}