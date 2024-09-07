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


warnings.filterwarnings("ignore")

class EmbeddingModelWithDropout:
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
        # noise_level = 0.1
        noise = torch.rand_like(input_tensor) * noise_level
        return input_tensor + noise

    def dropout_to_input(self,
                         input_tensor,
                         dropout_prob=0.1):
        dropout_layer = nn.Dropout(p=dropout_prob)
        masked_tensor = dropout_layer(input_tensor)
        return masked_tensor
    
    def get_output_with_dropout(self, text, noise=True):
    
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
            if noise:
                encoder_output = self.model.encoder.layer[i](self.dropout_to_input(prev_output))[0]
            else:
                encoder_output = self.model.encoder.layer[i](prev_output)[0]
            prev_output = encoder_output
            # output.append(encoder_output.detach().cpu())
        output_tensor = encoder_output.detach().cpu() # dim : (1, 512, 1024)
        # mean pooling
        output_tensor = output_tensor.mean(dim=1).squeeze(0) # dim : (1024)
        return output_tensor

    def get_output_with_dropout_middle_only(self, text, noise=True):
    
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
            if (len(self.model.encoder.layer) - 1 == i) and noise:
                encoder_output = self.model.encoder.layer[i](self.dropout_to_input(prev_output))[0]
            else:
                encoder_output = self.model.encoder.layer[i](prev_output)[0]
            prev_output = encoder_output
            # output.append(encoder_output.detach().cpu())
        output_tensor = encoder_output.detach().cpu() # dim : (1, 512, 1024)
        # mean pooling
        output_tensor = output_tensor.mean(dim=1).squeeze(0) # dim : (1024)
        return output_tensor


def load_vectorstore(
        db_root: str,
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
        db = DocumentAsQueryFAISS.load_local(db_root, embeddings=embeddings) 
        return db
    else:
        raise Exception(f'DB directory {db_root} is invalid.')


class RetrievalDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        csv_path: str = None,
    ):  
        # read data
        self.data_path = data_path
        with open(self.data_path) as f:
            corpus = [json.loads(i) for i in f.readlines()]
        
        if csv_path is not None:
            corpus_dict = {}
            for i in corpus:
                cid = i["_id"]
                corpus_dict[cid] = i
            df = pd.read_csv(csv_path)
            ids = df["corpus-id"].tolist()
            corpus = [corpus_dict[i] for i in ids] # random.sample([corpus_dict[i] for i in ids], 10)
            corpus = np.repeat(corpus, 10).tolist()

        self.dataset = corpus
        del corpus
        del corpus_dict
        
        print(f'Total {len(self.dataset)} data points') 
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
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
    
    def _get_context(self, inputs):
        retrieval_query = inputs["text"]
        docs = self.db.similarity_search_with_score(retrieval_query, **self.search_kwargs)
        retrieved = []
        for doc, score in docs:
            # record retrieval
            doc.metadata['document'] = doc.page_content
            # add result
            retrieved.append(doc.metadata)
        inputs["retrieval"] = retrieved
        return inputs
    
    def __call__(self, inputs):
        return self._get_context(inputs)   
    

class DocumentAsQueryFAISS(FAISS):

    def similarity_search_with_score(
            self,
            document_query: str,
            k: int = 4,
            filter = None,
            fetch_k: int = 20,
            **kwargs
        ):
        # embedding = self._embed_documents([document_query]) # List[List[float]]
        embedding = self.embedding_dropout_function.get_output_with_dropout_middle_only(("passage: " + document_query)).tolist() # List[List[float]]
        # print(embedding)
        # print("DEBUG:", len(embedding))
        # print("DEBUG:", len(embedding[0]))
        
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs
    
    # @classmethod
    # def load_local(
    #     self, 
    #     folder_path: str,
    #     embeddings: HuggingFaceEmbeddings,
    #     allow_dangerous_deserialization: bool = True
    # ):
    #     return FAISS.load_local(folder_path, 
    #                             embeddings=embeddings,
    #                             allow_dangerous_deserialization=allow_dangerous_deserialization
    #                             ) 
    
    @classmethod
    def load_local(
        cls, 
        folder_path: str,
        embeddings: HuggingFaceEmbeddings,
        allow_dangerous_deserialization: bool = True
    ):
        instance = super().load_local(
            folder_path, 
            embeddings=embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        
        instance.embedding_dropout_function = EmbeddingModelWithDropout("intfloat/multilingual-e5-large")
        return instance


        
def retrieve(
        # accelerator: Accelerator,
        dataset_name: str="trec-covid",
        data_root: str="/gallery_louvre/dayoon.ko/research/sds/src/datasets",
        save_root: str="results",
        db_faiss_dir: str="trec-covid",
        csv_path = "/gallery_louvre/dayoon.ko/research/sds/retrieval/results/multilingual-e5-large/trec-covid-fail.csv", # "results/multilingual-e5-large/trec-covid-fail.csv",
        top_k: int = 100,
        model_name: str = "intfloat/multilingual-e5-large"
    ):
    
    # load dataset
    data_path = f"{data_root}/{dataset_name}/corpus.jsonl"
    dataset = RetrievalDataset(data_path, csv_path=csv_path)    
    
    # make chain class
    retriever_db = load_vectorstore(db_faiss_dir, model_name=model_name)
    retrieval = Retrieval(retriever_db, search_kwargs={"k":top_k})
    
    # get dataset
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=dataset.collate_fn
                            )
    #dataloader = accelerator.prepare(dataloader)
    
    # path to save
    save_path = f'{save_root}/{dataset_name}.jsonl'
    if csv_path is not None:
        save_path = "results/e5/trec-covid-fail-d2d-retrieval.jsonl"
    print(f"Save to {save_path}")
    
    # execute inference
    results = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        output = retrieval(batch) 
        results.append(output)
        with open(save_path, 'a') as f:
            f.write(json.dumps(dict(output)) + '\n')


if __name__ == "__main__":
    fire.Fire(retrieve)
    
    