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


def get_count(
    model, 
    dropout,
    temp
):
    count_dict = defaultdict(dict)
    for dataset in "arguana climate-fever cqadupstack dbpedia-entity fiqa nfcorpus quora scidocs scifact trec-covid-v2 webis-touche2020".split():
        gradnorm_path_baseline = f"results/{model}/nq-train-result{dropout}-temp-{temp}"
        gradnorm_dict_baseline = get_gradnorms(gradnorm_path_baseline)
        num = len(gradnorm_dict_baseline.values())
        count_dict[dataset] = num
    return count_dict



def get_recall_mean(model, dataset, topk=""):
    pth = f"../retrieval/results/{model}/{dataset}/results.csv"
    df = pd.read_csv(pth)
    df = df.drop('Unnamed: 0', axis=1)
    df["corpus-id"] = df["corpus-id"].astype(str)
    scores = df[f"recall{topk}"].values 
    # recall_mean = sum(scores) / len(scores)
    num_queries = df["n-query"].values 
    recall_mean = sum([i * j for i, j in zip(scores, num_queries)]) / sum(num_queries)
    return recall_mean

def get_gradnorms(pth):
    lines = open(pth).readlines()
    lines_set = []
    for i, line in enumerate(lines[:-3]):
        if line.startswith("doc_id: ") and lines[i+3].startswith("{"):
            lines_set.append(lines[i:i+4])
    
    gradnorm_dict = defaultdict(list)
    for s in lines_set:
        sid = s[0].index("doc_id: ") + len("doc_id: ") 
        doc_id = s[0][sid:].strip()
        try:
            gradnorm = float(json.loads(s[-1].replace('\'', '"'))["grad_norm"])
        except:
            continue
        gradnorm_dict[doc_id].append(gradnorm)
    
    func = lambda x: sum(x) / len(x)
    gradnorm_dict = {k: func(v) for k, v in gradnorm_dict.items()}

    return gradnorm_dict


def model_eval(
    model,
    dropout="-0.02",
    randneg=False,
    topk=100
):
    temp = 0.01 if model=="multilingual-e5-large" else 0.05    
    count_dict = get_count(model, dropout, temp)
    
    temp = str(temp) + "-randneg" if randneg else temp
    topk = "" if topk == 100 else "-" + str(topk)
    
    # Find threshold quora cqadupstack
    datasets = "arguana climate-fever cqadupstack dbpedia-entity fiqa nfcorpus quora scidocs scifact trec-covid-v2 webis-touche2020" 
    results = []
    for dataset in datasets.split():
        #print(dataset, end=": ")
        recall_mean = get_recall_mean(model, dataset, topk)
        results.append(recall_mean)
        #print("recall mean:", round(recall_mean, 2))
    print(" & ".join([str(round(r*100, 1)) for r in results]), end=" & ")    
    avg = round(sum(results) / len(results) * 100, 2)
    print(avg)
    print("Avg:", avg)
        
if __name__ == "__main__":
    fire.Fire(model_eval) 