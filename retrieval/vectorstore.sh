#!/bin/sh
#dataset_name=$1
model_name=$1
repo=$2
batch_size=$3
dimension=$4


dataset_names='scidocs scifact webis-touche2020'
for dataset_name in $dataset_names; do
    echo "Model: $model_name"
    echo "Dataset: $dataset_name"  # remember quotes here
    echo "Dimension: $dimension"
    python vectorstore.py \
    --dataset_name ${dataset_name} \
    --db_faiss_dir ${model_name}/${dataset_name} \
    --batch_size ${batch_size} \
    --model_name ${repo}/${model_name} \
    --dimension ${dimension}
done