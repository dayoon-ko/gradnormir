data_root="eval_retrieval/retrieval/datasets"
dataset_name="scifact"
model_name="/gallery_louvre/sohyeon.kim/eval_retrieval/finetune/train/checkpoint-167"

python retriever.py \
   --dataset_name ${dataset_name} \
   --save_root "/gallery_louvre/sohyeon.kim/eval_retrieval/finetune/retrieval/results" \
   --db_faiss_dir vectorstore/${dataset_name}/ \
   --model_name ${model_name} \