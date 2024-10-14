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
from typing import Optional, Union
from glob import glob
import pandas as pd

warnings.filterwarnings("ignore")

def load_vectorstore(
        db_root: str,
        model_name: str = None
    ):
    
    # if vectorstore exists
    if os.path.exists(f'{db_root}/index.faiss'):
        embeddings = HuggingFaceEmbeddings(model_name=model_name, 
                                           model_kwargs={'device': 'cuda'},
                                           encode_kwargs={'batch_size': 2048,
                                                         #'show_progress_bar': False,
                                                         'device': 'cuda'
                                                         }
                                           )
        db = FAISS.load_local(db_root, embeddings=embeddings, allow_dangerous_deserialization=True) 
        return db
    else:
        raise Exception(f'DB directory {db_root} is invalid.')


class RetrievalDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        csv_path: str
    ):  
        # read data
        self.data_path = data_path
        with open(self.data_path) as f:
            dataset = [json.loads(i) for i in f.readlines()]
        self.filter(dataset, csv_path)
        print(f'Total {len(self.dataset)} data points') 
        
        # set baseline mode
        self.get_retrieval_query = self._get_question
            
        # set retrieval query
        self._prepare_retrieval_query()
        
    def filter(self, dataset, csv_path):
        qrels = pd.read_csv(csv_path)
        select_ids = set(qrels["query-id"].tolist())
        self.dataset = [i for i in dataset if i["_id"] in select_ids]
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
    def _prepare_retrieval_query(self):
        for item in self.dataset:
            item["question"] = self.get_retrieval_query(item)
    
    def _get_question(self, item):
        return item["text"]
    
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
        retrieval_query = inputs["question"]
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
        
        
         
def retrieve(
        # accelerator: Accelerator,
        dataset_name: str="climate-fever",
        data_root: str="/gallery_louvre/dayoon.ko/research/sds/src/datasets",
        sample_csv_path: str=None,
        save_root: str="results/multilingual-e5-large",
        db_faiss_dir: str=None,
        top_k: int = 100,
        model_name: str = "intfloat/multilingual-e5-large"
    ):
    
    # load dataset
    data_path = f"{data_root}/{dataset_name}/queries.jsonl"
    print("Data path:", data_path)
    dataset = RetrievalDataset(data_path, sample_csv_path)    
    
    # make chain class
    print(f"Load db from {db_faiss_dir}")
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
    
    