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
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
random.seed(0)

class DocumentLoader:

    def __init__(self, 
                 dataset_fn:str = "arguana"
                 ):
        self.dataset = self.load_dataset_file(dataset_fn)
        
    def load_dataset_file(self, dataset_fn):
        with open(dataset_fn) as f:
            json_file = [json.loads(i) for i in f.readlines()]
        return json_file

    def get_corpus(self, idx=0, data_type="train", num=5, randomly=False):

        corpus = [i["text"] for i in self.dataset if data_type in i["_id"]]
        if randomly:
            return random.sample(corpus, num)
        else:
            return corpus[idx:idx+num]
        


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

    def get_each_encoding_layer_output_adding_noise_continuously(self, text, add_noise=False,
                                                                 noise_level=0.1):
        
            # Output: [f1(x+e), f2(f1(x+e) + e), f3(f2(f1(x+e)+e)+e)
            #         where e is a gaussian noise.
        
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
        if add_noise:
            prev_output = self.add_noise_to_input(prev_output, noise_level=noise_level)
            
        # pass through each layer
        for i in range(len(self.model.encoder.layer)):
            encoder_output = self.model.encoder.layer[i](prev_output)[0]
            output.append(encoder_output.detach().cpu())
            
            # update prev_output for further forwarding
            prev_output = encoder_output
            if add_noise:
                prev_output = self.add_noise_to_input(prev_output, noise_level=noise_level)
                
        # reshape outputs
        output_tensor = torch.stack(output).transpose(1, 0)
        
        return output_tensor
    

    def get_each_encoding_layer_output_with_layerwise_noise(self, text, noise_level=0.1):
    
            # Output: [f1(x+e), f2(f1(x) + e), f3(f2(f(1)) + e), ...] 
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
        for i in range(len(self.model.encoder.layer)):
            # get output from noise-added input
            encoder_output = self.model.encoder.layer[i](self.add_noise_to_input(prev_output,
                                                                                 noise_level=noise_level))[0]
            output.append(encoder_output.detach().cpu())
            # update prev_output for further forwarding
            prev_output = self.model.encoder.layer[i](prev_output)[0]

        # reshape outputs
        output_tensor = torch.stack(output).transpose(1, 0)
        return output_tensor

        
    def get_each_att_values_with_layerwise_noise(self, text, embs, 
                                                 add_noise=False, 
                                                 noise_level=0.1):
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
        attns = []
        for i in range(len(self.model.encoder.layer)):
            if i == 0:
                input_embs = embedding_output
            else:
                input_embs = embs[i-1]
            if add_noise:
                input_embs = self.add_noise_to_input(input_embs, noise_level=noise_level)

            # calculate attention weights
            attn = self.model.encoder.layer[i].attention.self(input_embs)
            query = self.model.encoder.layer[i].attention.self.query(input_embs)
            key = self.model.encoder.layer[i].attention.self.key(input_embs)
            # value = self.model.encoder.layer[i].attention.self.value(input_embs)
            attention_weights = query.squeeze() @ key.squeeze().transpose(1,0) / query.shape[0]**(1/2)
            attention_weights = F.softmax(attention_weights.detach().cpu(), dim=-1).numpy()

            attns.append(attn[0].detach().cpu())
            
        attns_output = torch.stack(attns).transpose(1, 0)
        return attns_output


def umap_plot(embeddings, num_aug=5):
    embs = embeddings.transpose(1, 0)
    
    # loop over layer
    for i in range(len(embs)):
        embs_ith_layer = embs[i].detach().cpu()
        embs_ith_layer = embs_ith_layer.reshape(embs_ith_layer.shape[0], -1)
        
        # Apply UMAP to the combined tensor
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_result = reducer.fit_transform(embs_ith_layer)
        
        labels = np.array([i for i in range(0, embs_ith_layer.shape[0] //(num_aug + 1)) 
                            for _ in range(num_aug+1)])
        labels = np.expand_dims(labels, axis=1)
        
        indices = np.array([i for _ in range(0, embs_ith_layer.shape[0] //(num_aug + 1)) 
                            for i in range(num_aug+1)])
        indices = np.expand_dims(indices, axis=1)
        
        umap_result = np.concatenate([umap_result, labels, indices], axis=-1)
        data = pd.DataFrame(umap_result, columns=["x", "y", "color", "indices"])
        st.write(f"{i}th layer")
        st.scatter_chart(data,
                         x="x",
                         y="y",
                         color="color"
                         )
        



if __name__ == "__main__":
    
    # Get dataset name & model type 
    dataset_name = "arguana"
    model_name = "intfloat/multilingual-e5-large"
    
    # Load dataset
    dataset_fn = f"/gallery_louvre/dayoon.ko/research/sds/src/datasets/{dataset_name}/corpus.jsonl"   
    dl = DocumentLoader(dataset_fn)
    
    # Load model 
    embedder = EmbeddingModel(model_name)
    
    # Select documents 
    documents = {}
    idx = st.number_input(f"Start index of documents", 0, 200)
    num_doc = st.number_input(f"# Documents", 5, 10)
    for dtype in ["train", "val", "test"]:
        
        corpus = dl.get_corpus(idx=idx,
                                data_type=dtype,
                                num=num_doc,
                                randomly=False
                                )
        documents[dtype] = corpus
    
    # Get embeddings
    num_aug = st.number_input(f"# Times of noise addition", 5, 10)
    noise_level = float(st.text_input("Noise level", value="0.1"))
    all_texts = documents["train"] + documents["val"] + documents["test"]
    embs = []
    for text in all_texts:
        embs.append(embedder.get_each_encoding_layer_output_adding_noise_continuously(text, add_noise=False))
        for i in range(num_aug):
            embs.append(embedder.get_each_encoding_layer_output_with_layerwise_noise(text, noise_level=noise_level))
    embs = torch.cat(embs)
    
    # Umap plot
    umap_plot(embs)
    
    st.write(documents)
    
    
