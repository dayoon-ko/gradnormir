import math 
import json 
import umap
import torch 
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")
random.seed(0)        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-ds", default="nfcorpus")
    parser.add_argument("--fn", "-fn", default="corpus_selected.jsonl")
    parser.add_argument("--model_name", "-m", default="facebook/contriever")
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    return parser.parse_args()


class EmbeddingCollector:
    def __init__(self, model_name):
        #print("Model:", model_name)
        self.model_name = model_name 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, 
                                               output_attentions=True).to("cuda")
        self.set_pooling()
        
    def set_pooling(self):
        if "contriever" in self.model_name:
            self.pooling = "mean"
        elif "bge" in self.model_name:
            self.pooling = "mean"
        elif "e5" in self.model_name:
            self.pooling = "mean"
        elif "gte" in self.model_name:
            self.pooling = "cls"
            
    def get_num_layers(self):
        return len(self.model.encoder.layer)
    
    def forward(self, input_text):
        batch_dict = self.tokenizer(input_text, 
                                    max_length=512, 
                                    padding=True, 
                                    truncation=True, 
                                    return_tensors='pt')
        outputs = self.model(**batch_dict)
        return outputs

    def collect_embeedings(
            self, 
            text, 
            add_noise=False,
            noise_level=0.1, 
        ):
        # Output: [f1(x), f2(f1(x)), f3(f2(f1(x)))] where f_i is ith layer of the encoder.
        
        # tokenize
        encoded_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"].to("cuda")         
        attention_mask = encoded_input["attention_mask"].to("cuda")      
        
        # get embedding 
        embedding_output = self.model.embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=None,
                inputs_embeds=None,
                past_key_values_length=0,
            )
        # get each encoding layer output
        output = []
        prev_output = embedding_output
            
        # Pass through each layer
        for i in range(len(self.model.encoder.layer)):
            ith_hidden_states = self.model.encoder.layer[i](prev_output)[0]
            # Update prev_output for further forwarding
            prev_output = ith_hidden_states
            # Save embedding on ith layer
            masked_output = ith_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            output.append(masked_output.detach().cpu())
        
        # Reshape outputs
        output_tensor = torch.stack(output)
        if self.pooling == "cls":
            output_tensor = output_tensor[:,:,0, :]
        if self.pooling == "mean":
            output_tensor = output_tensor.sum(dim=-2)
            attention_mask = attention_mask.detach().cpu().sum(dim=-1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(-1)
            attention_mask = attention_mask.repeat(len(self.model.encoder.layer), 1, output_tensor.shape[-1])
            output_tensor = output_tensor / attention_mask        
        return output_tensor
    

class Corpus(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_dataset()

    def jsonl_to_json(self, jsonl_file):
        json_dict = {}
        for i in jsonl_file:
            json_dict[i["_id"]] = i["text"]
        return json_dict
    
    def load_dataset(self):
        with open(f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{self.args.dataset_name}/corpus_selected.jsonl") as f:
            self.dataset = [json.loads(i) for i in f.readlines()]
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
    @classmethod
    def collate_fn(cls, batch):
        ids = [i["_id"] for i in batch]
        texts = [i["text"] for i in batch]
        return ids, texts
        
    

def main(args):
    # Load dataset
    dataset = Corpus(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    
    # Load model 
    model = EmbeddingCollector(args.model_name)
    
    embeddings = []
    for ids, texts in tqdm(dataloader):
        outputs = model.collect_embeedings(texts)
        embeddings.append(outputs)
    embeddings = torch.cat(embeddings, dim=1)
    print(embeddings.shape)
    
    torch.save(embeddings, f"embeddings/{args.model_name.split('/')[-1]}/{args.dataset_name}.pt")
        
    
if __name__ == "__main__":
    args = get_args()
    main(args)
    
    