import torch 
import pandas as pd 
import json 
from glob import glob
from collections import defaultdict
import fire

from irmetrics.topk import recall

def get_count_dict(k=None):
    if k is None:
        return {
            "bge-large-en-v1.5": {"arguana": 126, "climate-fever": 140, "dbpedia-entity": 4230, "fiqa": 11970, "nfcorpus": 206, "quora": 486, "scidocs": 314, "scifact": 43, "trec-covid-v2": 2337, "webis-touche2020": 32, "cqadupstack": 9610},
            "contriever": {"arguana": 122, "climate-fever": 307, "dbpedia-entity": 7620, "fiqa": 7400, "nfcorpus": 1439, "quora": 3201, "scidocs": 1929, "scifact": 335, "trec-covid-v2": 5445, "webis-touche2020": 102, "cqadupstack": 11569},
            "multilingual-e5-large": {"arguana": 203, "climate-fever": 257, "dbpedia-entity": 1772, "fiqa": 13663, "nfcorpus": 1071, "quora": 2034, "scidocs": 1037, "scifact": 185, "trec-covid-v2": 12618, "webis-touche2020": 190, "cqadupstack": 19065},
            "gte-base": {"arguana": 1, "climate-fever": 89, "dbpedia-entity": 597, "fiqa": 1693, "nfcorpus": 54, "quora": 733, "scidocs": 439, "scifact": 4, "trec-covid-v2": 878, "webis-touche2020": 330, "cqadupstack": 519}
        }
    elif int(k) == 1000:
        return {
            "bge-large-en-v1.5": {'arguana': 122, 'climate-fever': 136, 'cqadupstack': 9461, 'dbpedia-entity': 4147, 'fiqa': 11756, 'nfcorpus': 197, 'quora': 467, 'scidocs': 297, 'scifact': 40, 'trec-covid-v2': 2157, 'webis-touche2020': 29},
            "contriever": {'arguana': 115, 'climate-fever': 299, 'cqadupstack': 11392, 'dbpedia-entity': 7536, 'fiqa': 7136, 'nfcorpus': 1392, 'quora': 3077, 'scidocs': 1885, 'scifact': 329, 'trec-covid-v2': 5224, 'webis-touche2020': 95},
            "multilingual-e5-large": {'arguana': 203, 'climate-fever': 257, 'cqadupstack': 19078, 'dbpedia-entity': 1727, 'fiqa': 13668, 'nfcorpus': 1072, 'quora': 2038, 'scidocs': 1038, 'scifact': 185, 'trec-covid-v2': 12625, 'webis-touche2020': 190},
            "gte-base": {'arguana': 3, 'climate-fever': 127, 'cqadupstack': 773, 'dbpedia-entity': 726, 'fiqa': 2047, 'nfcorpus': 80, 'quora': 910, 'scidocs': 552, 'scifact': 10, 'trec-covid-v2': 1271, 'webis-touche2020': 372}
        }
    elif int(k) == 2000:
        return {
            "bge-large-en-v1.5": {'arguana': 122, 'climate-fever': 137, 'cqadupstack': 9471, 'dbpedia-entity': 4151, 'fiqa': 11775, 'nfcorpus': 199, 'quora': 468, 'scidocs': 298, 'scifact': 40, 'trec-covid-v2': 2170, 'webis-touche2020': 29},
            "contriever": {'arguana': 118, 'climate-fever': 307, 'cqadupstack': 11486, 'dbpedia-entity': 7582, 'fiqa': 7289, 'nfcorpus': 1419, 'quora': 3150, 'scidocs': 1912, 'scifact': 334, 'trec-covid-v2': 5323, 'webis-touche2020': 98},
            "multilingual-e5-large": {'arguana': 199, 'climate-fever': 251, 'cqadupstack': 18965, 'dbpedia-entity': 1659, 'fiqa': 13546, 'nfcorpus': 1038, 'quora': 1977, 'scidocs': 1018, 'scifact': 179, 'trec-covid-v2': 12537, 'webis-touche2020': 180},
            "gte-base": {'arguana': 3, 'climate-fever': 119, 'cqadupstack': 722, 'dbpedia-entity': 711, 'fiqa': 1990, 'nfcorpus': 78, 'quora': 881, 'scidocs': 532, 'scifact': 9, 'trec-covid-v2': 1181, 'webis-touche2020': 366},
        }

def select_topk(
    model,
    dataset,
    topk = 100,
    k = None # NQ document samples
):
    # Set # of ood documents
    ood_topk = get_count_dict(k)[model][dataset]
    
    target = torch.load(f"embeddings/{model}/{dataset}-agg.pt", weights_only=False)
    indices = torch.topk(target, ood_topk).indices
    #print(len(indices), "are ood out of", len(target))
    
    # Select failed document ids
    with open(f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset}/corpus_selected.jsonl") as f:
        js = [json.loads(i) for i in f.readlines()]
    fail_doc_keys = set([js[i]["_id"] for i in indices])
    #print("Fail Detected: ", round(len(indices) / len(target) * 100, 2), "(", len(indices), "out of", len(target), ")")
    return fail_doc_keys



def get_recall_mean(model, dataset, selected_cids, topk=""):
    pth = f"../retrieval/results/{model}/{dataset}/results.csv"
    df = pd.read_csv(pth)
    df = df.drop('Unnamed: 0', axis=1)
    df["corpus-id"] = df["corpus-id"].astype(str)
    df = df[df["corpus-id"].isin(selected_cids)]
    scores = df[f"recall{topk}"].values 
    # recall_mean = sum(scores) / len(scores)
    num_queries = df["n-query"].values 
    recall_mean = sum([i * j for i, j in zip(scores, num_queries)]) / sum(num_queries)
    return recall_mean



def model_eval(
    model,
    topk=100,
    k=None # NQ document samples
):
    topk = "" if topk == 100 else "-" + str(topk)
    
    # Find threshold
    datasets = "arguana climate-fever cqadupstack dbpedia-entity fiqa nfcorpus quora scidocs scifact trec-covid-v2 webis-touche2020"
    results = []
    for dataset in datasets.split():
        #print(dataset)
        fail_cids = select_topk( # select_topk or compare_to_nq_score
            model,
            dataset,
            topk,
            k
        )
        recall_mean = get_recall_mean(model, dataset, fail_cids, topk)
        results.append(recall_mean)
        #print("recall mean:", round(recall_mean, 2))
    print(" & ".join([str(round(r*100, 2)) for r in results]), end=" & ")
    print(round(sum(results)/len(results) * 100, 2))

if __name__ == "__main__":
    fire.Fire(model_eval) 