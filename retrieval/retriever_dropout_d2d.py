from torch.utils.data import Dataset, DataLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from string import Template
import os
import fire
import json
import warnings
import torch
from glob import glob
import pandas as pd
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
import numpy as np
import torch.nn as nn
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.base import Docstore
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModel
import random
import torch.distributed as dist

warnings.filterwarnings("ignore")


class EmbeddingModelWithDropout:
    def __init__(self, model_name, layer_type):
        self.model_name = model_name 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, 
                                               output_attentions=True)
        self.model = model.to("cuda")
        self.layer_type = layer_type
        print(f"INIT with: {self.layer_type}")
    
    def forward(self, input_text):
        batch_dict = self.tokenizer(input_text, 
                                    max_length=512, 
                                    padding=True, 
                                    truncation=True, 
                                    return_tensors='pt')
        outputs = self.model(**batch_dict)
        return outputs


    def dropout_to_input(self,
                         input_tensor,
                         dropout_prob):
        dropout_layer = nn.Dropout(p=dropout_prob)
        masked_tensor = dropout_layer(input_tensor)
        return masked_tensor
    
    def get_output_with_dropout(self, text, dropout_prob, noise=True):
        # print(self.model.config )
    
            # Output: [f1(x), f2(f1(x)), f3(f2(f(1))), ..., f12(f11( ... ) + e), f13(f12(...)), ... ] 
            #        where e is a gaussian noise.
        # tokenize
        encoded_input = self.tokenizer(
            text,
            padding=True,
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
            if noise:
                if (self.layer_type == "all"):
                    encoder_output = self.model.encoder.layer[i](self.dropout_to_input(prev_output, dropout_prob))[0]
                elif (i == 0) and (self.layer_type == "first"):
                    encoder_output = self.model.encoder.layer[i](self.dropout_to_input(prev_output, dropout_prob))[0]
                elif ((len(self.model.encoder.layer) // 2) == i) and (self.layer_type == "middle"):
                    encoder_output = self.model.encoder.layer[i](self.dropout_to_input(prev_output, dropout_prob))[0]
                elif (len(self.model.encoder.layer) - 1 == i) and (self.layer_type == "final"):
                    encoder_output = self.model.encoder.layer[i](prev_output)[0]
                    encoder_output = self.dropout_to_input(encoder_output, dropout_prob)
                else: encoder_output = self.model.encoder.layer[i](prev_output)[0]
            else:
                encoder_output = self.model.encoder.layer[i](prev_output)[0]
            prev_output = encoder_output
            # output.append(encoder_output.detach().cpu())
        output_tensor = encoder_output.detach().cpu() # dim : (1, 512, 1024)
        # mean/cls pooling
        if "bge" in self.model_name:
            output_tensor = output_tensor[:, 0, :].squeeze(0) 
        else:
            output_tensor = output_tensor.mean(dim=1).squeeze(0) # dim : (1024)
        return output_tensor


def load_vectorstore(
        db_root: str,
        layer_type: str,
        model_name: str = None
    ):
    
    # if vectorstore exists 
    if os.path.exists(f'{db_root}/index.faiss'):
        embeddings = HuggingFaceEmbeddings(
                        model_name=model_name, 
                        model_kwargs={'device': 'cuda'},
                        encode_kwargs={
                            'batch_size': 2048,
                            'device': 'cuda'
                            },
                        show_progress=False
                        )
        db = DocumentAsQueryFAISS.load_local(db_root, model_name=model_name, layer_type=layer_type, embeddings=embeddings) 
        return db
    else:
        raise Exception(f'DB directory {db_root} is invalid.')


class RetrievalDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        csv_path: str,
    ):  
        # Load all document corpus
        self.data_path = data_path
        with open(self.data_path) as f:
            corpus = [json.loads(i) for i in f.readlines()]
        corpus = {i["_id"]: i for i in corpus}
        
        # Select documents of which ids are in csv file
        df = pd.read_csv(csv_path)
        selected_ids = df["corpus-id"].tolist()
        #selected_corpus = [corpus_dict[i] for i in ids]

        self.corpus = corpus
        self.dataset = selected_ids
            
        print(f'Total {len(self.dataset)} data points') 
    
    def __getitem__(self, idx):
        doc_id = self.dataset[idx]
        doc = self.corpus[doc_id]["text"]
        return doc_id, doc
    
    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        return batch[0]


class Retrieval:
    
    def __init__(
        self,
        db,
        search_kwargs = None,
    ):
        self.db = db
        self.search_kwargs = search_kwargs
        
    def _get_context(self, query):
        docs = self.db.similarity_search_with_score(query, **self.search_kwargs)
        retrieved_doc_ids = [doc.metadata["_id"] for doc, _ in docs]
        return retrieved_doc_ids
    
    def __call__(self, query):
        return self._get_context(query)   
        

class DocumentAsQueryFAISS(FAISS):
        
    def similarity_search_with_score(
            self,
            document_query: str,
            dropout_prob: float = 0.02,
            k: int = 4,
            filter = None,
            fetch_k: int = 20,
            **kwargs
        ):
        self.distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        self._normalize_L2 = True
        # embedding = self._embed_documents([document_query])
        embedding = self.embedding_dropout_function.get_output_with_dropout(document_query, dropout_prob).tolist() # List[List[float]] 

        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs
    
    @classmethod
    def load_local(
        cls, 
        folder_path: str,
        model_name: str,
        layer_type: str,
        embeddings: HuggingFaceEmbeddings,
        allow_dangerous_deserialization: bool = True
    ):
        cls.distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        cls._normalize_L2 = True
        # return FAISS.load_local(folder_path, 
        #                         embeddings=embeddings,
        #                         allow_dangerous_deserialization=allow_dangerous_deserialization
        #                         ) 
        instance = super().load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        instance.embedding_dropout_function = EmbeddingModelWithDropout(model_name=model_name, layer_type=layer_type)
        return instance
        
def retrieve(
        # accelerator: Accelerator,
        dataset_name: str="nfcorpus",
        data_root: str="/gallery_louvre/dayoon.ko/research/sds/src/datasets",
        top_k: int = 100,
        model_name: str = "BAAI/bge-m3", # "intfloat/multilingual-e5-large",
        layer_type: str = "final",
        dropout_prob: float = 0.01
    ):
    model_raw_name = model_name.split('/')[-1]
    csv_path= f"results/{model_raw_name}/{dataset_name}-n-query-mt-2.csv" # "/gallery_louvre/dayoon.ko/research/sds/retrieval/results/multilingual-e5-large/trec-covid-n-query-mt-2.csv",

    # Load dataset
    db_faiss_dir = f"vectorstore/{model_raw_name}/{dataset_name}"
    data_path = f"{data_root}/{dataset_name}/corpus.jsonl"
    dataset = RetrievalDataset(data_path, csv_path)    
    
    # Make a retrieval chain
    retriever_db = load_vectorstore(db_faiss_dir, layer_type, model_name=model_name)
    retrieval = Retrieval(retriever_db, search_kwargs={"k": top_k, "dropout_prob": dropout_prob})
    
    # Get a dataloader
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=dataset.collate_fn
                            )
    #dataloader = accelerator.prepare(dataloader)
    
    # Path to save
    if csv_path is not None:
        save_path = f"results/{model_raw_name}/{dataset_name}-n-query-mt-2-d2d-retrieval-{dropout_prob}.jsonl"
    print(f"Save to {save_path}")
        
    # Retrieve
    for _, (doc_id, doc) in tqdm(enumerate(dataloader), total=len(dataloader)):
        retrieved_doc_ids = retrieval(doc) 
        with open(save_path, 'a') as f:
            output = {"_id": doc_id, "retrieval": retrieved_doc_ids}
            f.write(json.dumps(dict(output)) + '\n')


if __name__ == "__main__":
    fire.Fire(retrieve)
    
    