import torch 
import pandas as pd 
import json 
from glob import glob
from collections import defaultdict
import fire
from pprint import pprint

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
    
    



def get_count(
    model, 
    dropout,
    temp,
    k=2001
):
    count_dict = defaultdict(dict)
    gradnorm_path_baseline = f"results/{model}/nq-train-result{dropout}-temp-{temp}"
    gradnorm_dict_baseline = get_gradnorms(gradnorm_path_baseline, k)
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
    
    print(count_dict)
    exit()
    return count_dict


def select_topk(
    model,
    dataset,
    ood_topk,
    temp, 
    dropout,
    topk = 100,
    num_pos = 16
):
    
    # Load gradnorm results
    gradnorm_path = f"results/{model}/{dataset}-result{dropout}-temp-{temp}"
    gradnorm_dict = get_gradnorms(gradnorm_path)
    gradnorm_path_8 = gradnorm_path.replace("/results", "/results_8")
    gradnorm_dict_8 = get_gradnorms(gradnorm_path_8)
    for k, v in gradnorm_dict.items():
        try:
            v.extend(gradnorm_dict_8[k])
            gradnorm_dict[k] = v
        except:
            continue
    for k, v in gradnorm_dict.items():
        gradnorm_dict[k] = sum(v[:num_pos]) / len(v[:num_pos])
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
    topk = 100,
    num_pos = 16
):    
    gradnorm_path = f"results/{model}/{dataset}-result{dropout}-temp-{temp}-0"
    gradnorm_dict = get_gradnorms(gradnorm_path)
    for i in ["1", "2", "3"]:
        gradnorm_path_ = gradnorm_path[:-1] + i
        gradnorm_dict_ = get_gradnorms(gradnorm_path_)
        for k, v in gradnorm_dict.items():
            try:
                v.extend(gradnorm_dict_8[k])
                gradnorm_dict[k] = v
            except:
                continue
    for k, v in gradnorm_dict.items():
        gradnorm_dict[k] = sum(v[:num_pos]) / len(v[:num_pos])
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

def get_gradnorms(pth, k=None):
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
        if k is not None and len(gradnorm_dict) == k:
            break
    #func = lambda x: sum(x) / len(x)
    #gradnorm_dict = {k: func(v) for k, v in gradnorm_dict.items() if len(v) > 1}

    return gradnorm_dict


def model_eval(
    model,
    dropout="-0.02",
    num_pos=16,
    randneg=False,
    topk=100,
    k:int=None # num of nq documents / None: 3000
):
    
    temp = 0.05 if model=="multilingual-e5-large" else 0.05    
    #count_dict = get_count(model, dropout, temp)
    count_dict = get_count_dict(k)[model] # select same count
    
    temp = str(temp) + "-randneg" if randneg else temp
    topk = "" if topk == 100 else "-" + str(topk)
        
    for num_pos in [1,2,4,8,16]:
        # Find threshold quora cqadupstack
        datasets = "climate-fever cqadupstack dbpedia-entity fiqa nfcorpus quora scidocs scifact trec-covid-v2 webis-touche2020" #fiqa cqadupstack
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
                    topk,
                    num_pos
                )
            else:
                fail_cids = select_topk( # select_topk or compare_to_nq_score
                    model,
                    dataset,
                    count_dict[dataset],
                    temp, 
                    dropout,
                    topk,
                    num_pos
                )
            recall_mean = get_recall_mean(model, dataset, fail_cids, topk)
            results.append(recall_mean)
            #print("recall mean:", round(recall_mean, 2))
        print(" & ".join([str(round(r*100, 2)) for r in results]), end=" & ")    
        avg = round((sum(results) + 0.9951)/ (1+len(results)) * 100, 2)
        print(avg)
        #print("Avg:", avg)
        
if __name__ == "__main__":
    fire.Fire(model_eval) 