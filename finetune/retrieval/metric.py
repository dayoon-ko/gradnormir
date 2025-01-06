import json 
import pandas as pd
from glob import glob
from collections import defaultdict
import fire
import os
from irmetrics.topk import (
    ndcg, 
    recall,
    precision
)
from pprint import pprint 
import numpy as np

METRIC_FUNCS = {
    "ndcg": ndcg,
    "recall": recall,
    "precision": precision
}

def load_jsonl(pth):
    with open(pth) as f: 
        jsonl = [json.loads(i) for i in f.readlines()]
    results = {i["_id"]: [r["_id"] for r in i["retrieval"]] for i in jsonl}
    return results


def load_qrels(root, dataset_name, score_type="binary"):
    qrels_glob = f"{root}/{dataset_name}/qrels/*.tsv"
    qrels = pd.DataFrame()
    print(qrels_glob)
    for qrels_fn in glob(qrels_glob):
        qrels_ = pd.read_csv(qrels_fn, sep="\t")
        qrels_ = qrels_[qrels_["score"] >= 1]
        
        query_id_type = qrels_["query-id"].dtype
        if query_id_type == 'int64' or query_id_type == 'float64':
            qrels = pd.concat([qrels, qrels_], axis=0)
        else:
            qrels = pd.concat([qrels, qrels_], axis=1)
            
    qrels = qrels.set_index("query-id")
    
    if score_type == "max_score":
        qrels_dict = defaultdict(dict)
        for qid, row in qrels.iterrows():
            cid = row["corpus-id"]
            score = row["score"]
            qrels_dict[str(qid)][str(cid)] = score
            
    elif score_type == "binary":
        qrels_dict = defaultdict(list)
        for qid, row in qrels.iterrows():
            cid = row["corpus-id"]
            qrels_dict[str(qid)].append(str(cid))
            
    return qrels_dict 


def cal_metrics(
    output_fn: str = None,
    dataset_name: str = "trec-covid-v2",
    qrels_root: str = "/gallery_louvre/dayoon.ko/research/sds/src/datasets",
    save_path: str = None,
    is_baseline: bool = False
):  
    
    os.makedirs(f'{save_path}', exist_ok=True)
    if not is_baseline:
        output = load_jsonl(output_fn)
    else:
        output = load_jsonl(f"/gallery_louvre/dayoon.ko/research/eval_retrieval/retrieval/results/{model_name}/{dataset_name}.jsonl")
    qrels_dict = load_qrels(qrels_root, dataset_name)
    
    outputs = {met: {at: 0 for at in [5, 20, 50, 100]} for met in ["recall", "ndcg", "precision"]}
    
    #nan_found = False  
    
    for qid, y_pred in output.items():
        y_true = qrels_dict[qid]
        # print("y_true", y_true)
        # print("y_pred", y_pred[:10])
        for met, func in METRIC_FUNCS.items():
            for top_k in [5, 20, 50, 100]:
                score = 0
                if met == "ndcg":
                    y_pred_ = [1 if i in y_true else 0 for i in y_pred][:top_k]
                    y_true_ = [1 for i in y_true]
                    score = func(y_true_, y_pred_)
                else:
                    score = func(y_true, y_pred[:top_k])
                    
                # if not nan_found and np.isnan(score):
                #     print(f"NaN detected!")
                #     print(f"Query ID: {qid}")
                #     print(f"Metric: {met}")
                #     print(f"Top-K: {top_k}")
                #     print(f"y_true: {y_true}")
                #     print(f"y_pred: {y_pred[:top_k]}")
                #     print(f"y_pred_: {y_pred_ if met == 'ndcg' else 'N/A'}")
                #     print(f"y_true_: {y_true_ if met == 'ndcg' else 'N/A'}")
                #     nan_found = True 
                
                outputs[met][top_k] += score / len(output) * 100
                
    eval_results_path = os.path.join(save_path, "eval_results.json")
    with open(eval_results_path, 'w') as f:
        json.dump(outputs, f, indent=4)
        print(f"Saved evaluation results to {eval_results_path}")
    
    pprint(outputs)
    
    
    
    
if __name__ == "__main__":
    fire.Fire(cal_metrics)
    

'''    
trec-covid-v2 baseline

{'ndcg': {5: nan,
          20: 39.54147401492146,
          50: 39.29452321578953,
          100: 39.076034664211456},
 'precision': {5: 29.0,
               20: 26.84999999999999,
               50: 22.89999999999999,
               100: 19.68},
 'recall': {5: 0.4155259406648404,
            20: 1.4789374296171145,
            50: 3.0280262290856697,
            100: 5.069386654131679}}
}
'''
'''
Save to /gallery_louvre/dayoon.ko/research/eval_retrieval/finetune/outputs/trec-covid-v2/ood/lr-5e-05-grad-2-epc-/trec-covid-v2.jsonl
100%|████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 27.67it/s]
Finish to retrieve
Start to calculate metric
/gallery_louvre/dayoon.ko/research/sds/src/datasets/trec-covid-v2/qrels/*.tsv
File names: /gallery_louvre/dayoon.ko/research/sds/src/datasets/trec-covid-v2/qrels/test.tsv
/gallery_louvre/dayoon.ko/anaconda3/envs/dynamicer/lib/python3.10/site-packages/irmetrics/topk.py:270: RuntimeWarning: invalid value encountered in divide
  return dcg_score(relevant, k) / idcg
{'ndcg': {5: nan,
          20: 38.80758220568555,
          50: 38.60797179080297,
          100: 38.5680727091644},
 'precision': {5: 28.200000000000003, 20: 25.5, 50: 21.66, 100: 18.3},
 'recall': {5: 0.395648632197258,
            20: 1.3742173460828413,
            50: 2.7701680288539263,
            100: 4.52368627059123}}
'''