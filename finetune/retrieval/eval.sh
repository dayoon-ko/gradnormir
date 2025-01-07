output_fn="/gallery_louvre/sohyeon.kim/eval_retrieval/finetune/retrieval/results/arguana.jsonl"
dataset_name="arguana"
qrels_root="/gallery_louvre/sohyeon.kim/eval_retrieval/retrieval/datasets"

python metric.py \
    --output_fn ${output_fn} \
    --dataset_name ${dataset_name} \
    --qrels_root ${qrels_root} \
    --save_path /gallery_louvre/sohyeon.kim/eval_retrieval/finetune/retrieval/eval/${dataset_name} \
    --is_baseline False