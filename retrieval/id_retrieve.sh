#!/bin/sh
dataset_name=$1
model_name=$2
repo=$3

sr 1 24 -q low python id_retriever.py \
--dataset_name ${dataset_name} \
--save_root results/${model_name} \
--sample_csv_path results/${model_name}/${dataset_name}.csv \
--db_faiss_dir vectorstore/${model_name}/${dataset_name} \
--model_name ${repo}/${model_name}