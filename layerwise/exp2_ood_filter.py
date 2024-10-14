import torch 
import pandas as pd 
import json 
from glob import glob
from collections import defaultdict
import fire

from irmetrics.topk import recall

count = {
    "bge-large-en-v1.5": {
        "arguana": 126,
        "climate-fever": 140,
        "dbpedia-entity": 4230,
        "fiqa": 11970,
        "nfcorpus": 206,
        "quora": 486,
        "scidocs": 314,
        "scifact": 43,
        "trec-covid-v2": 2337,
        "webis-touche2020": 32,
        "cqadupstack": 9610,
    },
    "contriever": {
        "arguana": 122,
        "climate-fever": 307,
        "dbpedia-entity": 7620,
        "fiqa": 7400,
        "nfcorpus": 1439,
        "quora": 3201,
        "scidocs": 1929,
        "scifact": 335,
        "trec-covid-v2": 5445,
        "webis-touche2020": 102,
        "cqadupstack": 11569,
    },
    "multilingual-e5-large": {
        "arguana": 203,
        "climate-fever": 257,
        "dbpedia-entity": 1772,
        "fiqa": 13663,
        "nfcorpus": 1071,
        "quora": 2034,
        "scidocs": 1037,
        "scifact": 185,
        "trec-covid-v2": 12618,
        "webis-touche2020": 190,
        "cqadupstack": 19065,
    },
    "gte-base": {
        "arguana": 1,
        "climate-fever": 89,
        "dbpedia-entity": 597,
        "fiqa": 1693,
        "nfcorpus": 54,
        "quora": 733,
        "scidocs": 439,
        "scifact": 4,
        "trec-covid-v2": 878,
        "webis-touche2020": 330,
        "cqadupstack": 519,
    }
}
def compare_to_nq_score(
    model,
    dataset,
    topk = 100
):
    
    # Load gradnorm results
    ref = torch.load(f"embeddings/{model}/nq-train-agg.pt", weights_only=False)
    thrs = ref.sort().values[int(len(ref) * 0.95)]

    target = torch.load(f"embeddings/{model}/{dataset}-agg.pt", weights_only=False)
    indices = torch.where(target > thrs)[0]
    print(len(indices), "are ood out of", len(target))
    
    # Select failed document ids
    with open(f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset}/corpus_selected.jsonl") as f:
        js = [json.loads(i) for i in f.readlines()]
    fail_doc_keys = set([js[i]["_id"] for i in indices])
    print("Fail Detected: ", round(len(indices) / len(target) * 100, 2), "(", len(indices), "out of", len(target), ")")

    # Load qrels
    glob_dir = f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset}/qrels/*.tsv"
    qrels_all = pd.DataFrame()
    for pth in glob(glob_dir):
        qrels = pd.read_csv(pth, sep="\t")
        qrels["query-id"] = qrels["query-id"].astype(str)
        qrels["corpus-id"] = qrels["corpus-id"].astype(str)
        qrels_all = pd.concat([qrels_all, qrels])

    # Bipartition qrels
    failed_queries = qrels_all[qrels_all["corpus-id"].isin(fail_doc_keys)]["query-id"].unique()
    #succed_queries = qrels_all[~qrels_all["corpus-id"].isin(fail_doc_keys)]["query-id"].unique()
    succed_queries = [i for i in qrels_all["query-id"].unique() if i not in failed_queries]
    print(f"Queries: All {len(qrels_all['query-id'].unique())} \tFailed {len(failed_queries)} \tSucceeded: {len(succed_queries)}")
    
    # Load retrieval results
    with open(f"../retrieval/results/{model}/{dataset}.jsonl") as f:
        js = [json.loads(i) for i in f.readlines()]
        js = {i["_id"]: [ret["_id"] for ret in i["retrieval"]] for i in js}

    # Bipartition retrieval results
    failed_js, succed_js = {}, {}
    for qid, ret in js.items():
        target = failed_js if qid in failed_queries else succed_js
        target[qid] = ret
    
    if len(succed_js) == 0:
        return None
    
    func = lambda x: round(sum(x) / len(x) * 100, 2)
    qrels_all = qrels_all.set_index("query-id")
    
    scores = func([recall(qrels_all.loc[k]["corpus-id"], v[:topk]) 
                       #if len(set(qrels_all.loc[k]["corpus-id"]) & set(v[:topk])) > 0 else 0
                        for k, v in js.items()])
    scores_fail = func([recall(qrels_all.loc[k]["corpus-id"], v[:topk]) 
                        #if len(set(qrels_all.loc[k]["corpus-id"]) & set(v[:topk])) > 0 else 0
                        for k, v in failed_js.items()])
    scores_succ = func([recall(qrels_all.loc[k]["corpus-id"], v[:topk]) 
                        #if len(set(qrels_all.loc[k]["corpus-id"]) & set(v[:topk])) > 0 else 0
                        for k, v in succed_js.items()])

    print("Avg:", scores, "\tFail:", scores_fail, "\tSucc:", scores_succ)
    print(f"Avg: {scores} ({len(js)}) \tFail: {scores_fail} ({len(failed_js)}) \tSucc: {scores_succ} ({len(succed_js)})")
    return {"avg": scores, "fail": scores_fail, "succ": scores_succ}


def select_topk(
    model,
    dataset,
    topk = 100
):

    target = torch.load(f"embeddings/{model}/{dataset}-agg.pt", weights_only=False)
    ood_topk = count[model][dataset]
    indices = torch.topk(target, ood_topk).indices
    print(len(indices), "are ood out of", len(target))
    
    # Select failed document ids
    with open(f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset}/corpus_selected.jsonl") as f:
        js = [json.loads(i) for i in f.readlines()]
    fail_doc_keys = set([js[i]["_id"] for i in indices])
    print("Fail Detected: ", round(len(indices) / len(target) * 100, 2), "(", len(indices), "out of", len(target), ")")

    # Load qrels
    glob_dir = f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset}/qrels/*.tsv"
    qrels_all = pd.DataFrame()
    for pth in glob(glob_dir):
        qrels = pd.read_csv(pth, sep="\t")
        qrels = qrels[qrels["score"] >= 1]
        qrels["query-id"] = qrels["query-id"].astype(str)
        qrels["corpus-id"] = qrels["corpus-id"].astype(str)
        qrels_all = pd.concat([qrels_all, qrels])

    # Bipartition qrels
    failed_queries = qrels_all[qrels_all["corpus-id"].isin(fail_doc_keys)]["query-id"].unique()
    #succed_queries = qrels_all[~qrels_all["corpus-id"].isin(fail_doc_keys)]["query-id"].unique()
    succed_queries = [i for i in qrels_all["query-id"].unique() if i not in failed_queries]
    print(f"Queries: All {len(qrels_all['query-id'].unique())} \tFailed {len(failed_queries)} \tSucceeded: {len(succed_queries)}")
    
    # Load retrieval results
    with open(f"../retrieval/results/{model}/{dataset}.jsonl") as f:
        js = [json.loads(i) for i in f.readlines()]
        js = {i["_id"]: [ret["_id"] for ret in i["retrieval"]] for i in js}

    # Bipartition retrieval results
    failed_js, succed_js = {}, {}
    for qid, ret in js.items():
        target = failed_js if qid in failed_queries else succed_js
        target[qid] = ret
    
    if len(succed_js) == 0:
        return None
    
    func = lambda x: round(sum(x) / len(x) * 100, 2)
    qrels_all = qrels_all.set_index("query-id")
    
    scores = func([recall(qrels_all.loc[k]["corpus-id"], v[:topk]) 
                       #if len(set(qrels_all.loc[k]["corpus-id"]) & set(v[:topk])) > 0 else 0
                        for k, v in js.items()])
    scores_fail = func([recall(qrels_all.loc[k]["corpus-id"], v[:topk]) 
                        #if len(set(qrels_all.loc[k]["corpus-id"]) & set(v[:topk])) > 0 else 0
                        for k, v in failed_js.items()])
    scores_succ = func([recall(qrels_all.loc[k]["corpus-id"], v[:topk]) 
                        #if len(set(qrels_all.loc[k]["corpus-id"]) & set(v[:topk])) > 0 else 0
                        for k, v in succed_js.items()])

    print("Avg:", scores, "\tFail:", scores_fail, "\tSucc:", scores_succ)
    print(f"Avg: {scores} ({len(js)}) \tFail: {scores_fail} ({len(failed_js)}) \tSucc: {scores_succ} ({len(succed_js)})")
    return {"avg": scores, "fail": scores_fail, "succ": scores_succ}


def model_eval(
    model,
    topk=100
):
    # Find threshold
    datasets = "arguana climate-fever cqadupstack dbpedia-entity fiqa nfcorpus quora scidocs scifact trec-covid-v2 webis-touche2020"
    results = {"avg": {}, "fail": {}, "succ": {}}
    for dataset in datasets.split():
        print(dataset)
        result = select_topk( # select_topk or compare_to_nq_score
            model,
            dataset,
            topk
        )
        if result is None:
            continue
        for k, v in result.items():
            results[k][dataset] = v
        print()

    print("Latex")
    for t in ["avg", "fail", "succ"]:
        print(t)
        print(" & ".join([str(results[t][dataset]) for dataset in datasets.split() if dataset in results[t]]))
        
if __name__ == "__main__":
    fire.Fire(model_eval) 