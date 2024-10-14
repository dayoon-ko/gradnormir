import os 
import json 
import fire
import pandas as pd 
from glob import glob 
from typing import Optional
from collections import defaultdict
from pprint import pprint 
import random 
random.seed(0)

def load_jsonl(pth):
    output = []
    count = 0
    with open(pth) as f:
        #return [json.loads(i) for i in f.readlines()]
        lines = f.readlines()
        for _, l in enumerate(lines):
            try:
                output.append(json.loads(l))
            except:
                count += 1
                continue 
        print(f"Wrong: {count}")
    return output 

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

def load_qrels(model_name, dataset_name):
    qrels = pd.read_csv(f"results/{model_name}/{dataset_name}/results.csv")
    qrels = qrels.drop('Unnamed: 0', axis=1)
    qrels = qrels.astype({"corpus-id": str})
    qrels = qrels.set_index("corpus-id")
    return qrels

def load_retrieval_dict(dataset_name, model_name):
    with open(f"results/{model_name}/{dataset_name}.jsonl") as f:
        results = [json.loads(i) for i in f.readlines()]
    retrieval_dict = defaultdict(list)
    for res in results:
        qid = res["_id"]
        for retrieval in res["retrieval"]:
            doc_id = retrieval["_id"]
            retrieval_dict[doc_id].append(qid)
    return retrieval_dict
    

def get_ret_map(
    pth, 
    dropout, 
    num_positives, 
    num_negatives_per_positive,
    positive_start_idx
):
    
    def filter_data(res):
        for i in res:
            i["retrieval"] = [j for j in i["retrieval"] if i["_id"] != j]
        return res
    
    if not str(dropout).startswith("-"):
        dropout=""

    pth0 = pth.replace(".csv", f"-d2d-retrieval.jsonl")
    res0 = load_jsonl(pth0)
    res0 = filter_data(res0)
    res0 = {i["_id"]: i["retrieval"] for i in res0}
    pth1 = pth.replace(".csv", f"-d2d-retrieval{dropout}.jsonl")
    res1 = load_jsonl(pth1)
    res1 = filter_data(res1)
    res1 = {i["_id"]: i["retrieval"] for i in res1}
    pth2 = pth.replace(".csv", f"-d2d2d-retrieval{dropout}.jsonl")
    res2 = load_jsonl(pth2)
    res2 = filter_data(res2)
    res2 = {i["_id"]: i["retrieval"][1:] for i in res2}
    results = {}
    for qid, ret_ids in res1.items():
        selected_rets = {}
        for ret_id in ret_ids[positive_start_idx:num_positives]:
            try:
                ret2_ids = res2[ret_id]
            except:
                print(ret_id)
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
    root: str = "/gallery_louvre/dayoon.ko/research/sds/src/datasets",
    dataset_name:str = None,
    model_name:str = None,
    dropout: str = "",
    num_positives:int = 16,
    num_negatives_per_positive:int = 4,
    positive_start_idx: int = 0,
    random_neg: bool = False,
    chunk_idx: int = None,
):  
    # Check whether existing
    if not str(dropout).startswith("-"):
        dropout = ""
    if random_neg:
        save_path = f"../gradnorm/datasets/{model_name}/{dataset_name}{dropout}-randneg.json"
        if chunk_idx is not None:
            save_path = save_path.replace(".json", f"-{chunk_idx}.json")
    else:
        save_path = f"../gradnorm/datasets/{model_name}/{dataset_name}{dropout}.json"
    if positive_start_idx > 0:
        save_path = save_path.replace("datasets", f"datasets_8")
    print(save_path)
    
    # Load queries, corpus, and qrels
    qrels = load_qrels(model_name, dataset_name)
    corpus_dict = load_dataset_dict(root, dataset_name, "corpus_selected")
    random_negatives = None
    if random_neg:
        random_negatives = random.sample(list(corpus_dict.values()), num_negatives_per_positive)
    print(f"Dataset {dataset_name} is loaded!")
    
    # Load retrieval results
    retrieval_map = get_ret_map(
                        f"results/{model_name}/{dataset_name}/results.csv", 
                        dropout,
                        num_positives, 
                        num_negatives_per_positive,
                        positive_start_idx
                    )
    print(f"Retrieval results of {model_name} is loaded!")
    
    # Generate dataset
    dataset = []
    for query_id, rets1 in retrieval_map.items():
        for pos_id, neg_ids in rets1.items():
            query_meta = qrels.loc[query_id]
            query = corpus_dict[query_id]
            positive = [corpus_dict[pos_id]]
            if random_neg:
                negatives = random_negatives
            else:
                negatives = [corpus_dict[idx] for idx in neg_ids if idx in corpus_dict]
            dataset.append({
                "_id": query_id,
                "recall": float(query_meta["recall"]),
                #"precision": query_meta["precision"],
                #"f1": query_meta["f1"],
                "query": query,
                "pos": positive,
                "neg": negatives,
            }) 
    
    # Save the positive & negative document dataset
    with open(save_path, "w") as f:
        print(len(dataset))
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    fire.Fire(main)