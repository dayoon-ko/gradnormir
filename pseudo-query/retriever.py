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
        save_path: str
    ):  
        # read data
        with open(data_path) as f:
            self.dataset = [json.loads(i) for i in f.readlines()]
        print(f'Total: {len(self.dataset)} data points')
        if os.path.exists(save_path):
            with open(save_path) as f:
                self.dataset = self.dataset[len(f.readlines()):]
        print(f'To do: {len(self.dataset)} data points')
        
    
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
        
    def _retrieve(self, qid, query):
        docs = self.db.similarity_search_with_score(query, **self.search_kwargs)
        doc_ids = [str(doc.metadata["_id"]) for doc, _ in docs]
        try:
            rank = doc_ids.index(str(qid))
        except:
            rank = -1
        return rank
    
    def __call__(self, inputs):
        rank = self._retrieve(inputs["_id"], inputs["text"])   
        inputs["rank"] = rank
        return inputs
        
        
def retrieve(
        # accelerator: Accelerator,
        dataset_name: str="climate-fever",
        data_root: str="/gallery_louvre/dayoon.ko/research/sds/src/datasets",
        save_root: str="results",
        db_faiss_dir: str=None,
        top_k: int = 100,
        model_name: str = "intfloat/multilingual-e5-large"
    ):
    
    # load dataset
    data_path = f"{data_root}/{dataset_name}/queries_generated.jsonl"
    print("Load queries from:", data_path)
    save_path = f"{save_root}/{model_name.split('/')[-1]}/{dataset_name}.jsonl"
    print("Save results in:", save_path)
    dataset = RetrievalDataset(data_path, save_path)    
    if len(dataset) == 0:
        return
    
    # make chain class
    if db_faiss_dir is None:
        db_faiss_dir = f"/gallery_louvre/dayoon.ko/research/eval_retrieval/retrieval/vectorstore/{model_name.split('/')[-1]}/{dataset_name}"
    print(f"Load db from {db_faiss_dir}")
    retriever_db = load_vectorstore(db_faiss_dir, model_name=model_name)
    retrieval = Retrieval(retriever_db, search_kwargs={"k":top_k})
    
    # get dataset
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=dataset.collate_fn,
                            )
    
            
    # execute inference
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        out = retrieval(batch)
        with open(save_path, 'a') as f:
            f.write(json.dumps(dict(out)) + '\n')


if __name__ == "__main__":
    fire.Fire(retrieve)
    
    