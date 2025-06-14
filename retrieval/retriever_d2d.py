from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import fire
import json
import logging
import pandas as pd
import numpy as np
import torch.nn as nn
from utils import load_vectorstore, Retrieval

logger = logging.getLogger("__main__")
logger.setLevel(logging.INFO)

class RetrievalDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        save_path: str
    ):  
        # Load all document corpus and filter done
        self.load_dataset(data_path)
        self.filter_done(save_path)
        logger.info(f'Total {len(self.dataset)} data points') 
        
    def load_dataset(self, data_path):
        with open(data_path) as f:
            corpus = [json.loads(i) for i in f.readlines()]
        self.corpus = {i["_id"]: i for i in corpus}
    
    def filter_done(self, save_path):
        if os.path.exists(save_path):
            with open(save_path) as f:
                already_ids = set([json.loads(i)["_id"] for i in f.readlines()])
        else: 
            already_ids = []
        selected_ids = [i for i in self.corpus 
                        if i not in already_ids]
        self.dataset = selected_ids
    
    def __getitem__(self, idx):
        doc_id = self.dataset[idx]
        doc = self.corpus[doc_id]["text"]
        return doc_id, doc
    
    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        return batch[0]

        
def retrieve(
        data_root: str,
        dataset_name: str,
        save_root: str,
        db_faiss_dir: str,
        model_name: str,
        top_k: int=100,
    ):
    
    # Load dataset
    data_path = f"{data_root}/{dataset_name}/corpus_selected.jsonl"
    if not os.path.exists(data_path):
        raise Exception(f"There is no file {data_path}")
    save_path = f"{save_root}/d2d-retrieval.jsonl"
    
    logger.info(f"Load dataset from {data_path}")
    logger.info(f"Save results to {save_path}")
    
    # Load dataset
    dataset = RetrievalDataset(
                data_path=data_path, 
                save_path=save_path
            )    
    
    # Get a dataloader
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=dataset.collate_fn
                            )
    
    # Make a retrieval chain
    retriever_db = load_vectorstore(
                        model_name=model_name,
                        db_faiss_dir=db_faiss_dir
                    )
    
    # Make a retrieval chain
    retrieval = Retrieval(
                        retriever_db, 
                        search_kwargs={
                            "k": top_k
                        })
        
    # Retrieve
    for _, (doc_id, doc) in tqdm(enumerate(dataloader), total=len(dataloader)):
        retrieved_doc_ids = retrieval(doc) 
        with open(save_path, 'a') as f:
            output = {"_id": doc_id, "retrieval": retrieved_doc_ids}
            f.write(json.dumps(output) + '\n')


if __name__ == "__main__":
    fire.Fire(retrieve)
    
    