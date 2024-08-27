import json 
import umap
import torch 
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
random.seed(0)

class EmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, 
                                               output_attentions=True)
        self.model = model.to("cuda")
    
    def forward(self, input_text):
        batch_dict = self.tokenizer(input_text, 
                                    max_length=512, 
                                    padding=True, 
                                    truncation=True, 
                                    return_tensors='pt')
        outputs = self.model(**batch_dict)
        return outputs

    def add_noise_to_input(self, 
                           input_tensor, 
                           noise_level=1):
        noise = torch.rand_like(input_tensor) * noise_level
        return input_tensor + noise

    def dropout_to_input(self,
                         input_tensor,
                         dropout_prob=0.5):
        dropout_layer = nn.Dropout(p=dropout_prob)
        masked_tensor = dropout_layer(input_tensor)
        return masked_tensor


    def get_last_output_with_middle_layer_noise(self, text, noise_level=0.1, noise_type = None):
    
            # Output: [f1(x), f2(f1(x)), f3(f2(f(1))), ..., f12(f11( ... ) + e), f13(f12(...)), ... ] 
            #        where e is a gaussian noise.

        # tokenize
        encoded_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        encoded_input = {
                key: val.to("cuda") for key, val in encoded_input.items()
            }
        # get embedding 
        embedding_output = self.model.embeddings(
                input_ids=encoded_input["input_ids"],
                position_ids=None,
                token_type_ids=None,
                inputs_embeds=None,
                past_key_values_length=0,
            )
        # get each encoding layer output
        output = []
        prev_output = embedding_output
        # output.append(embedding_output.detach().cpu())
        # prev_output = self.add_noise_to_input(prev_output, noise_level=noise_level)
        for i in range(len(self.model.encoder.layer)):
            if noise_type and (len(self.model.encoder.layer) // 2 == i):
                if noise_type == "Gaussian":
                    encoder_output = self.model.encoder.layer[i](self.add_noise_to_input(prev_output))[0]
                elif noise_type == "Dropout":
                    encoder_output = self.model.encoder.layer[i](self.dropout_to_input(prev_output))[0]
                else:
                    NotImplementedError
            else:
                encoder_output = self.model.encoder.layer[i](prev_output)[0]
            prev_output = encoder_output
            # output.append(encoder_output.detach().cpu())
        output_tensor = encoder_output.detach().cpu() # dim : (1, 512, 1024)
        # mean pooling
        output_tensor = output_tensor.mean(dim=1).squeeze(0) # dim : (1024)
        return output_tensor


def jsonl_to_json(jsonl_file):
    json_dict = {}
    for i in jsonl_file:
        json_dict[i["_id"]] = i 
    return json_dict
        

def load_dataset(dataset_name, dtype="train"):
    qrels = pd.read_csv(f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset_name}/qrels/{dtype}.tsv", sep="\t")
    with open(f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset_name}/corpus.jsonl") as f:
        corpus = jsonl_to_json([json.loads(i) for i in f.readlines()])
    with open(f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset_name}/queries.jsonl") as f:
        queries = jsonl_to_json([json.loads(i) for i in f.readlines()])
    return {
            "qrels": qrels, 
            "corpus": corpus,
            "queries": queries
           }


if __name__ == "__main__":
    
    # Get dataset name & model type 
    id_dataset_name = "fever"
    ood_dataset_name = "arguana"
    model_name = "intfloat/multilingual-e5-large"
    
    idx = 0
    num_doc = 5

    # Load dataset
    id_dataset = load_dataset(id_dataset_name, dtype="train")
    id_corpus_ids = id_dataset["qrels"]["corpus-id"].tolist()[idx : idx + num_doc]
    id_corpus = [("passage: " + id_dataset["corpus"][id]["text"]) for id in id_corpus_ids]
    
    ood_dataset = load_dataset(ood_dataset_name, dtype="test")
    ood_corpus_ids = ood_dataset["qrels"]["corpus-id"].tolist()[idx : idx + num_doc]
    ood_corpus = [("passage: " + ood_dataset["corpus"][id]["text"]) for id in ood_corpus_ids]   
    
    # Load model
    embedder = EmbeddingModel(model_name)

    # Select documents
    documents = {"id": id_corpus, "ood": ood_corpus}

    # Get embeddings
    num_aug = 5
    noise_level = 0.1
    all_texts = documents["id"] + documents["ood"]

    for noise_type in [None, "Gaussian", "Dropout"]:
        print(f"Noise_type: {str(noise_type)}")
        # embs = []
        for idx, text in enumerate(all_texts):
            embs = []
            for i in range(num_aug):
                embs.append(embedder.get_last_output_with_middle_layer_noise(text, noise_type=noise_type))
            vectors_tensor = torch.stack(embs)

            cosine_sim_matrix = torch.nn.functional.cosine_similarity(
                vectors_tensor.unsqueeze(1), vectors_tensor.unsqueeze(0), dim=2
            )

            n = len(embs)
            upper_tri_indices = np.triu_indices(n, k=1)
            cosine_sim_values = cosine_sim_matrix[upper_tri_indices]

            average_cosine_similarity = cosine_sim_values.mean().item()
            stdev_cosine_similarity = cosine_sim_values.std().item()

            print(f"Text Index: {idx}")
            print(f"Average Cosine Similarity: {average_cosine_similarity:.5f}")
            # print(f"Standard Deviation of Cosine Similarity: {stdev_cosine_similarity}")
            print()

