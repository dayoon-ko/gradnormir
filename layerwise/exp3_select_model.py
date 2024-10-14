import torch 
import pandas as pd 
import json 
from glob import glob
from collections import defaultdict
import fire

from irmetrics.topk import recall


def select_topk(
    model,
    dataset,
    topk
):
    # Set # of ood documents
    #thrs = means[model]

    target = torch.load(f"embeddings/{model}/{dataset}-agg.pt", weights_only=False)
    count = sum(target > -0.9999)
    print(model, ":", count)
    return count
    


def model_eval(topk=100):
    # Find threshold quora cqadupstack
    datasets = "arguana climate-fever cqadupstack dbpedia-entity fiqa nfcorpus quora scidocs scifact trec-covid-v2 webis-touche2020".split()
    models = ["bge-large-en-v1.5", "contriever", "gte-base", "multilingual-e5-large"]
    #datasets = "quora"
    results = {"avg": {}, "fail": {}, "succ": {}}
    for dataset in datasets:
        print(dataset)
        scores = []
        for model in models:
            score = select_topk( # select_topk or compare_to_nq_score
                model,
                dataset,
                topk
            )
            scores.append((model, score))
        scores = sorted(scores, key=lambda x:x[1])
        print("Best:", scores[0][0])
        print()
            
if __name__ == "__main__":
    fire.Fire(model_eval) 