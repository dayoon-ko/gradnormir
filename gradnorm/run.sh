#!/bin/sh
model_name=$1
repo=$2
pooling=$3
dropout=$4
temperature=$5
port=$6
dataset_name=$7
data_root=$8

# > ${model_name}/${dataset_name}-result${dropout}-temp-${temperature}

echo "Model: $model_name"
echo "Dataset: $dataset_name"  # remember quotes here
echo "Dropout: $dropout"
echo "Temperature: $temperature"
echo "Pooling: $pooling"

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:${port} \
    run.py \
    --model_name_or_path ${repo}/${model_name} \
    --output_dir outputs \
    --train_data ${data_root}/${dataset_name}-${dropout}.json \
    --same_task_within_batch True \
    --sentence_pooling_method ${pooling} \
    --do_train False \
    --temperature ${temperature} \
    --logging_pth results/${model_name}/${dataset_name}-${dropout}