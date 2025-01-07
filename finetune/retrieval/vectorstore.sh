dataset_name="webis-touche2020"
model_name="/gallery_louvre/sohyeon.kim/eval_retrieval/finetune/train/checkpoint-230"

python vectorstore.py \
   --dataset_name ${dataset_name} \
   --glob_dir "corpus_selected.jsonl" \
   --db_faiss_dir vectorstore/${dataset_name} \
   --batch_size 256 \
   --model_name ${model_name}
