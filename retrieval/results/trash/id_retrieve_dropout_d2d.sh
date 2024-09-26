#!/bin/sh
dataset_name=$1
model_name=$2
repo=$3
top_k=$4
dropout_prob=$5

echo "dataset_name: $1"
echo "model_name: $2"
echo "repo: $3"
echo "top_k: $4"
echo "dropout_ratio: $5"

sr 1 24 -q low python retriever_dropout_d2d.py \
--dataset_name ${dataset_name} \
--model_name ${repo}/${model_name} \
--csv_path results/${model_name}/${dataset_name}.csv \
--top_k ${top_k} \
--dropout_prob ${dropout_prob}
