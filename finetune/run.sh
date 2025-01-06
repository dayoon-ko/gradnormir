#!/bin/sh
model_name=$1
dataset_name=$2

# > ${dataset_name}

echo "Model: $model_name"
echo "Dataset: $dataset_name"  # remember quotes here
# echo "Dropout: $dropout"
# echo "Temperature: $temperature"
# echo "Pooling: $pooling"

torchrun \
    run.py \
    --model_name_or_path ${model_name} \
    --train_data /gallery_louvre/sohyeon.kim/eval_retrieval/retrieval/datasets/${dataset_name}/corpus_selected.jsonl \
    --output_dir /gallery_louvre/sohyeon.kim/eval_retrieval/finetune/train-dataset/${dataset_name}/ \
    --same_task_within_batch True \
    --sentence_pooling_method mean \
    --per_device_train_batch_size 4 \
    --do_train True \
    --report_to "wandb"