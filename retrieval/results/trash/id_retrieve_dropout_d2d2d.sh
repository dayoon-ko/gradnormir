#!/bin/sh
dataset_name= $1-train
model_name=$2
repo=$3
dropout_prob=$4
csv_path= results/${model_name}/${dataset_name}.csv

echo $dataset_name
echo $csv_path

sr 1 24 -q low python retriever_dropout_d2d2d.py \
--dataset_name ${dataset_name} \
--db_faiss_dir vectorstore/${model_name}/${dataset_name} \
--csv_path $csv_path \
--model_name ${repo}/${model_name} \
--dropout_prob ${dropout_prob}