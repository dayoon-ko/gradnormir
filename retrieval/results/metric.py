import json 
import pandas as pd
from glob import glob
from collections import defaultdict
import fire
from irmetrics.topk import (
    ndcg, 
    recall,
    precision
)
from pprint import pprint 

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
    for qrels_fn in glob(qrels_glob):
        qrels_ = pd.read_csv(qrels_fn, sep="\t")
        qrels_ = qrels_[qrels_["score"] >= 1]
        qrels_["query-id"] = qrels_["query-id"].astype(str)
        qrels_["corpus-id"] = qrels_["corpus-id"].astype(str)
        qrels = pd.concat([qrels, qrels_])
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
    model_name: str = None,
    output_fn: str = None,
    dataset_name: str = "trec-covid-v2",
    qrels_root: str = "/gallery_louvre/dayoon.ko/research/sds/src/datasets",
    is_baseline: bool = False
):  
    if not is_baseline:
        output = load_jsonl(output_fn)
    else:
        output = load_jsonl(f"/gallery_louvre/dayoon.ko/research/eval_retrieval/retrieval/results/{model_name}/{dataset_name}.jsonl")
    qrels_dict = load_qrels(qrels_root, dataset_name)
    
    outputs = {met: {at: 0 for at in [5, 20, 50, 100]} for met in ["recall", "ndcg", "precision"]}
    
    for qid, y_pred in output.items():
        y_true = qrels_dict[qid]
        for met, func in METRIC_FUNCS.items():
            if len(y_true) == 0:
                continue
            for top_k in [5, 20, 50, 100]:
                score = 0
                if met == "ndcg":
                    y_pred_ = [1 if i in y_true else 0 for i in y_pred][:top_k]
                    if sum(y_pred_) == 0:
                        continue
                    y_true_ = [1 for i in y_true]
                    score = func(y_true_, y_pred_)
                else:
                    score = func(y_true, y_pred[:top_k])
                outputs[met][top_k] += score / len(output) * 100
    pprint(outputs) 
    return(outputs)
    


def eval_all(
    dataset_name: str = "trec-covid-v2",
    qrels_root: str = "/gallery_louvre/dayoon.ko/research/sds/src/datasets",
    is_baseline: bool = False
):  
    results_all = {}
    for dataset_name in "fiqa trec-covid-v2 nfcorpus dbpedia-entity climate-fever webis-touche2020 scidocs quora cqadupstack".split():
        results_dset = {}
        for model_name in "bge-large-en-v1.5 multilingual-e5-large gte-base contriever".split():
            output_fn = f"{model_name}/{dataset_name}.jsonl"
            results = cal_metrics(
                model_name=model_name, 
                output_fn=output_fn, 
                dataset_name=dataset_name
            )
            results_dset[model_name] = results
        results_all[dataset_name] = results_dset
    
    with open("q2d_metric.json", "w") as f:
        json.dump(results_all, f, indent=2)


def compare():
    with open("q2d_metric.json") as f:
        js = json.load(f)
    for dataset, result in js.items():
        print(dataset)
        for metric in ["recall", "precision", "ndcg"]:
            for topk in [5, 20, 50, 100]:
                order = []
                for model in ["bge-large-en-v1.5", "multilingual-e5-large", "contriever", "gte-base"]:
                    order.append([model, result[model][metric][str(topk)]])
                order = sorted(order, key=lambda x: x[1], reverse=True)
                print(metric, topk, end = ": ")
                for i in order:
                    print(i[0][:5], round(i[1]), end=" / ")
                print()
            print("\n")
        print("\n\n\n")
            
            

def compare_dset():
    with open("q2d_metric.json") as f:
        js = json.load(f)
    
    for model in ["bge-large-en-v1.5", "multilingual-e5-large", "contriever", "gte-base"]:
        print(model)
        for metric in ["recall", "precision", "ndcg"]:
            for topk in [5, 20, 50, 100]:
                order = []                
                for dataset, result in js.items():
                    order.append([dataset, result[model][metric][str(topk)]])
                order = sorted(order, key=lambda x: x[1], reverse=True)
                print(metric, topk, end = ": ")
                for i in order:
                    print(i[0][:5], end=" / ")
                print()
            print("\n")
        print("\n\n\n")
            
    
    
if __name__ == "__main__":
    #fire.Fire(eval_all)
    fire.Fire(compare)
    

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