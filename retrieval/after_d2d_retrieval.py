import os 
import json 
import fire
import pandas as pd 
from glob import glob 
from typing import Optional
from collections import defaultdict
from pprint import pprint 
import random 
from tqdm import tqdm
random.seed(0)


def load_jsonl(pth):
    output = []
    with open(pth) as f:
        lines = f.readlines()
    for _, l in enumerate(lines):
        try:
            output.append(json.loads(l))
        except:
            continue 
    return output 


def load_dataset_dict(
    data_root,
    dataset_name,
    file_name
):
    assert file_name in ["queries", "corpus_selected"]
    with open(f"{data_root}/{dataset_name}/{file_name}.jsonl") as f:
        jsonl = [json.loads(i) for i in f.readlines()]
        jsonl_dict = {}
        for q in jsonl:
            jsonl_dict[q["_id"]] = q["text"]
    return jsonl_dict


def load_qrels(data_root, dataset_name):
    qrels_all = pd.DataFrame()
    for qrels_fn in glob(f"{data_root}/{dataset_name}/qrels/*.tsv"):
        qrels = pd.read_csv(qrels_fn, sep="\t")
        qrels = qrels[qrels["score"] >= 1].sort_values("corpus-id")
        qrels_all = pd.concat([qrels_all, qrels])
    return qrels_all
    

def get_ret_map(
    save_root, 
    dropout, 
    num_positives, 
    num_negatives_per_positive
):
    
    # Filter the qurery from the retrieval results
    def filter_data(res):
        for i in res:
            i["retrieval"] = [j for j in i["retrieval"] if i["_id"] != j]
        return res
    
    # Load jsonl file and get results
    def get_result(pth):
        res = load_jsonl(pth)
        res = filter_data(res)
        if "d2d2d" not in pth:
            res = {i["_id"]: i["retrieval"] for i in res}
        else:
            res = {i["_id"]: i["retrieval"][1:] for i in res}
        return res
    
    # Load d2d, d2d w/ dropout, and d2d2d results
    res0 = get_result(f"{save_root}/d2d-retrieval.jsonl")
    res1 = get_result(f"{save_root}/d2d-retrieval-{dropout}.jsonl")
    res2 = get_result(f"{save_root}/d2d2d-retrieval-{dropout}.jsonl")

    # Get result map of positives and according negatives
    results = {}
    for qid, ret_ids in res1.items():
        selected_rets = {}
        for ret_id in ret_ids[:num_positives]:
            try:
                ret2_ids = res2[ret_id]
            except:
                continue
            ret2_ids_selected = [i for i in ret2_ids if i not in res0[qid] and i != qid]
            ret2_ids_selected = ret2_ids_selected[:num_negatives_per_positive]
            if len(ret2_ids_selected) < num_negatives_per_positive:
                continue
            selected_rets[ret_id] = ret2_ids_selected
            if len(selected_rets) == num_positives:
                break
        results[qid] = selected_rets
    return results
        

def main(
    data_root: str,
    dataset_name: str,
    save_root: str,
    dropout: float,
    num_positives: int = 16,
    num_negatives_per_positive: int = 4
):  
    # Check save root
    if not os.path.exists(save_root):
        os.path.makedirs(save_root, exist_ok=True)
    save_path = f"{save_root}/{dataset_name}-{dropout}.json"
    print(f"Save pos & neg dataset to {save_path}")
    
    # Load queries, corpus, and qrels
    qrels = load_qrels(data_root, dataset_name)
    corpus_dict = load_dataset_dict(data_root, dataset_name, "corpus_selected")
    print(f"Dataset {dataset_name} is loaded!")
    
    # Load retrieval results
    retrieval_map = get_ret_map(
                        save_root, 
                        dropout,
                        num_positives, 
                        num_negatives_per_positive
                    )
    
    # Load q2d retrievals
    with open(f"{save_root}/q2d-retrieval.jsonl") as f:
        q2d_rets = [json.loads(i) for i in f.readlines()]
        q2d_rets = {i["_id"]: i["retrieval"] for i in q2d_rets}
    
    # Generate dataset
    dataset = []
    for doc_query_id, pos_neg_map in tqdm(retrieval_map.items()):
        for pos_id, neg_ids in pos_neg_map.items():
            gold_queries = qrels[qrels["corpus-id"]==doc_query_id]["query-id"].values
            gold_ret_queries = [qid for qid in gold_queries if doc_query_id in q2d_rets[qid]]
            recall = len(gold_ret_queries) / len(gold_queries)
            doc_query = corpus_dict[doc_query_id]
            positive = [corpus_dict[pos_id]]
            negatives = [corpus_dict[idx] for idx in neg_ids if idx in corpus_dict]
            dataset.append({
                "_id": doc_query_id,
                "recall": recall,
                "n-query": len(gold_queries),
                "query": doc_query,
                "pos": positive,
                "neg": negatives,
            }) 
    
    # Save the positive & negative document dataset
    with open(save_path, "w") as f:
        print(f"Total {len(dataset)} points are generated")
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    fire.Fire(main)