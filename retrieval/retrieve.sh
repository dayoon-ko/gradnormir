#!/bin/sh
model_name=$1
repo=$2
dataset_name=$3

echo "dataset_name: $dataset_name"
echo "model_name: $model_name"

sr 1 24 -q low python retriever.py \
--dataset_name ${dataset_name} \
--save_root results/${model_name} \
--db_faiss_dir vectorstore/${model_name}/${dataset_name} \
--model_name ${repo}/${model_name}