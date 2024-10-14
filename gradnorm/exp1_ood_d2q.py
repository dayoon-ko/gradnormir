import torch 
import pandas as pd 
import json 
from glob import glob
from collections import defaultdict
import fire
from pprint import pprint

from irmetrics.topk import recall

def get_count_dict():
    return {
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
    gradnorm_path_baseline = f"results/{model}/nq-train-result{dropout}-temp-{temp}"
    gradnorm_dict_baseline = get_gradnorms(gradnorm_path_baseline)
    thrs = sum(gradnorm_dict_baseline.values()) / len(gradnorm_dict_baseline.values())
    
    for dataset in "arguana climate-fever cqadupstack dbpedia-entity fiqa nfcorpus quora scidocs scifact trec-covid-v2 webis-touche2020".split():
        # Load gradnorm results
        gradnorm_path = f"results/{model}/{dataset}-result{dropout}-temp-{temp}"
        gradnorm_dict = get_gradnorms(gradnorm_path)
        gradnorm_path_8 = gradnorm_path.replace("/results", "/results_8")
        gradnorm_dict_8 = get_gradnorms(gradnorm_path_8)
        for k, v in gradnorm_dict.items():
            try:
                gradnorm_dict[k] = (gradnorm_dict[k] + gradnorm_dict_8[k]) / 2
            except:
                continue
    
        # Select failed document ids
        fail_doc_keys = set([k for k, v in gradnorm_dict.items() if v > thrs])
        count_dict[dataset] = len(fail_doc_keys)
            
    return count_dict


def select_topk(
    model,
    dataset,
    ood_topk,
    temp, 
    dropout,
    topk = 100
):
    
    # Load gradnorm results
    gradnorm_path = f"results/{model}/{dataset}-result{dropout}-temp-{temp}"
    gradnorm_dict = get_gradnorms(gradnorm_path)
    gradnorm_path_8 = gradnorm_path.replace("/results", "/results_8")
    gradnorm_dict_8 = get_gradnorms(gradnorm_path_8)
    for k, v in gradnorm_dict.items():
        try:
            gradnorm_dict[k] = (gradnorm_dict[k] + gradnorm_dict_8[k]) / 2
        except:
            continue
    target = gradnorm_dict.items()
    # target = [(i, r) for i, r in enumerate(target)]
    fail_doc_keys = sorted(target, key=lambda x: x[1], reverse=True)[:ood_topk]
    fail_doc_keys = [i for i, _ in fail_doc_keys]
    #print(len(fail_doc_keys), "are ood out of", len(target))
    return fail_doc_keys        

def select_topk_cqa(
    model,
    dataset,
    ood_topk,
    temp, 
    dropout,
    topk = 100
):
    
    gradnorm_path = f"results/{model}/{dataset}-result{dropout}-temp-{temp}-0"
    gradnorm_dict = get_gradnorms(gradnorm_path)
    for i in ["1", "2", "3"]:
        gradnorm_path_ = gradnorm_path[:-1] + i
        gradnorm_dict_ = get_gradnorms(gradnorm_path_)
        for k, v in gradnorm_dict.items():
            try:
                gradnorm_dict[k] = gradnorm_dict[k] + gradnorm_dict_[k]
            except:
                continue
    for k, v in gradnorm_dict.items():
        gradnorm_dict[k] /= 4
    target = gradnorm_dict.items()
    fail_doc_keys = sorted(target, key=lambda x: x[1], reverse=True)[:ood_topk]
    fail_doc_keys = [i for i, _ in fail_doc_keys]
    #print(len(fail_doc_keys), "are ood out of", len(target))
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
    recall_mean = sum([ i * j for i, j in zip(scores, num_queries)]) / sum(num_queries)
    #recall_mean = sum([ (1-i) * j for i, j in zip(scores, num_queries)]) / sum(num_queries)
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
    #count_dict = get_count(model, dropout, temp)
    count_dict = get_count_dict()[model] # select same count
    
    temp = str(temp) + "-randneg" if randneg else temp
    topk = "" if topk == 100 else "-" + str(topk)
    
    # Find threshold quora cqadupstack
    datasets = "arguana climate-fever cqadupstack dbpedia-entity fiqa nfcorpus quora scidocs scifact trec-covid-v2 webis-touche2020" #fiqa cqadupstack
    results = []
    for dataset in datasets.split():
        #print(dataset, end=": ")
        if dataset == "cqadupstack" and randneg:
            fail_cids = select_topk_cqa( # select_topk or compare_to_nq_score
                model,
                dataset,
                count_dict[dataset],
                temp, 
                dropout,
                topk
            )
        else:
            fail_cids = select_topk( # select_topk or compare_to_nq_score
                model,
                dataset,
                count_dict[dataset],
                temp, 
                dropout,
                topk
            )
        recall_mean = get_recall_mean(model, dataset, fail_cids, topk)
        results.append(recall_mean)
        #print("recall mean:", round(recall_mean, 2))
    print(" & ".join([str(round(r*100, 2)) for r in results]))    
    print("Avg:", round(sum(results) / len(results) * 100, 2))
        
if __name__ == "__main__":
    fire.Fire(model_eval) 