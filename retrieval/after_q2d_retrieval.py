import os 
import json 
import fire
import pandas as pd 
from glob import glob 
from typing import Optional
from collections import defaultdict

def load_dataset_dict(
    root,
    dataset_name,
    file_name
):
    assert file_name in ["queries", "corpus_selected"]
    with open(f"{root}/{dataset_name}/{file_name}.jsonl") as f:
        jsonl = [json.loads(i) for i in f.readlines()]
        jsonl_dict = {}
        for q in jsonl:
            jsonl_dict[q["_id"]] = q["text"]
    return jsonl_dict

def load_qrels(root, dataset_name):
    qrels_all = pd.DataFrame()
    for qrels_fn in glob(f"{root}/{dataset_name}/qrels/*.tsv"):
        qrels = pd.read_csv(qrels_fn, sep="\t")
        qrels = qrels[qrels["score"] >= 1].sort_values("corpus-id")
        qrels_all = pd.concat([qrels_all, qrels])
    return qrels_all

def load_retrieval_dict(dataset_name, model_name, top_k):
    with open(f"results/{model_name}/{dataset_name}.jsonl") as f:
        results = [json.loads(i) for i in f.readlines()]
    retrieval_dict = defaultdict(list)
    for res in results:
        qid = res["_id"]
        for retrieval in res["retrieval"][:top_k]:
            doc_id = retrieval["_id"]
            retrieval_dict[doc_id].append(qid)
    return retrieval_dict
    
def generate_gt_dict(qrels):
    # Generate GT dictionary
    gt_dict = {}
    for k in qrels["corpus-id"].unique():
        qrels_ = qrels[qrels["corpus-id"] == k]
        v = [str(i) for i in qrels_["query-id"].tolist()]
        gt_dict[str(k)] = v
    return gt_dict

def main(
    root: str = "/gallery_louvre/dayoon.ko/research/sds/src/datasets",
    dataset_name:str = None,
    model_name:str = None,
):
    # Load queries, corpus, and qrels
    qrels = load_qrels(root, dataset_name)
    queries_dict = load_dataset_dict(root, dataset_name, "queries")
    corpus_dict = load_dataset_dict(root, dataset_name, "corpus_selected")
    gt_dict = generate_gt_dict(qrels)
    print(f"Dataset {dataset_name} is loaded!")
    
    # Load retrieval results
    retrieval_dict_100 = load_retrieval_dict(dataset_name, model_name, top_k=100)
    retrieval_dict_20 = load_retrieval_dict(dataset_name, model_name, top_k=20)
    retrieval_dict_50 = load_retrieval_dict(dataset_name, model_name, top_k=50)
    print(f"Retrieval results of {model_name} is loaded!")
    
    # Calculate the metric
    corpus_ids = []
    num_gts = []
    recall = [] # over # gt
    recall_20 = []
    recall_50 = []
    precision = [] # over # rets
    precision_20 = []
    precision_50 = []
    
    for k, gt_v in gt_dict.items():
        # gt_v: document 가 retrieval 되어야하는 모든 쿼리 / v: document 가 실제 retrieval 된 쿼리
        gt_v = [str(i) for i in gt_v]
        corpus_ids.append(k)
        num_gts.append(len(gt_v))
        
        v = retrieval_dict_100[k] if k in retrieval_dict_100 else [] 
        tp_t = len(set(gt_v).intersection(set(v))) / len(gt_v) # True Positive / # True
        tp_p = len(set(gt_v).intersection(set(v))) / len(v) if len(v) > 0 else 0 # True Positive / # Positive
        recall.append(tp_t)
        precision.append(tp_p)  
        
        v = retrieval_dict_20[k] if k in retrieval_dict_20 else [] 
        tp_t = len(set(gt_v).intersection(set(v))) / len(gt_v) # True Positive / # True
        tp_p = len(set(gt_v).intersection(set(v))) / len(v) if len(v) > 0 else 0 # True Positive / # Positive
        recall_20.append(tp_t)
        precision_20.append(tp_p)
        
        v = retrieval_dict_50[k] if k in retrieval_dict_50 else [] 
        tp_t = len(set(gt_v).intersection(set(v))) / len(gt_v) # True Positive / # True
        tp_p = len(set(gt_v).intersection(set(v))) / len(v) if len(v) > 0 else 0 # True Positive / # Positive
        recall_50.append(tp_t)
        precision_50.append(tp_p)
    
    # Generate dataset
    data = {
        "corpus-id": corpus_ids, 
        "n-query": num_gts,
        "recall": recall,
        "recall-20": recall_20,
        "recall-50": recall_50,
        "precision": precision,
        "precision-20": precision_20,
        "precision-50": precision_50
       }
    df = pd.DataFrame(data)
    df = df[df["n-query"] >= 1]
    df["f1"] = [0 if i < 0.001 and j < 0.001 
                else 2 * i * j / (i + j) 
                for i, j in zip(df.recall, df.precision)]
    
    # Save the metadata
    save_root = f"results/{model_name}/{dataset_name}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    df.to_csv(f"{save_root}/results.csv")
    

if __name__ == "__main__":
    fire.Fire(main)