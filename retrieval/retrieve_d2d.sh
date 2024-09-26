#!/bin/sh
dataset_name=$1
model_name=$2
repo=$3
top_k=$4

echo "dataset_name: $1"
echo "model_name: $2"

csv_path="results/${model_name}/${dataset_name}-n-query-mt-2.csv"

echo "csv_path: $csv_path"

sr 1 24 -q low python retriever_d2d.py \
--dataset_name ${dataset_name} \
--db_faiss_dir vectorstore/${model_name}/${dataset_name} \
--model_name ${repo}/${model_name} \
--csv_path results/${model_name}/${dataset_name}-n-query-mt-2.csv \
--top_k ${top_k}
