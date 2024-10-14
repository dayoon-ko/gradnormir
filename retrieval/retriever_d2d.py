from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from torch.utils.data import Dataset, DataLoader
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
        embeddings = HuggingFaceEmbeddings(
                        model_name=model_name, 
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={
                            'batch_size': 2048,
                            'device': 'cpu'
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
        csv_path: str,
        save_path: str = None
    ):  
        # Load all document corpus
        self.data_path = data_path
        with open(self.data_path) as f:
            corpus = [json.loads(i) for i in f.readlines()]
        corpus = {i["_id"]: i for i in corpus}
        
        # Select documents of which ids are in csv file
        df = pd.read_csv(csv_path)
        selected_ids = [str(i) for i in df["corpus-id"].tolist()]
        #selected_corpus = [corpus_dict[i] for i in ids]
        
        if os.path.exists(save_path):
            with open(save_path) as f:
                already_ids = set([json.loads(i)["_id"] for i in f.readlines()])
            selected_ids = [i for i in selected_ids if i not in already_ids]
        
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
            k: int = 4,
            filter = None,
            fetch_k: int = 20,
            **kwargs
        ):
        self.distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        self._normalize_L2 = True
        embedding = self._embed_documents([document_query])
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
        embeddings: HuggingFaceEmbeddings,
        allow_dangerous_deserialization: bool = True
    ):
        cls.distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        cls._normalize_L2 = True
        return FAISS.load_local(folder_path, 
                                embeddings=embeddings,
                                allow_dangerous_deserialization=allow_dangerous_deserialization
                                ) 
        
def retrieve(
        # accelerator: Accelerator,
        dataset_name: str="trec-covid",
        data_root: str="/gallery_louvre/dayoon.ko/research/sds/src/datasets",
        csv_path: str=None,
        db_faiss_dir: str="trec-covid",
        top_k: int = 100,
        model_name: str = "intfloat/multilingual-e5-large"
    ):
    
    # Load dataset
    data_path = f"{data_root}/{dataset_name}/corpus_selected.jsonl"
    
    # Path to save
    if csv_path is not None:
        save_path = csv_path.replace(".csv", "-d2d-retrieval.jsonl")
    if top_k != 100:
        save_path = save_path.replace(".jsonl", f"-{top_k}.jsonl")
    print(f"Save to {save_path}")
    
    dataset = RetrievalDataset(
                data_path, 
                csv_path, 
                save_path=save_path
            )    
    
    # Make a retrieval chain
    retriever_db = load_vectorstore(db_faiss_dir, model_name=model_name)
    retrieval = Retrieval(retriever_db, search_kwargs={"k":top_k})
    
    # Get a dataloader
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=dataset.collate_fn
                            )
    #dataloader = accelerator.prepare(dataloader)
        
    # Retrieve
    for _, (doc_id, doc) in tqdm(enumerate(dataloader), total=len(dataloader)):
        retrieved_doc_ids = retrieval(doc) 
        with open(save_path, 'a') as f:
            output = {"_id": doc_id, "retrieval": retrieved_doc_ids}
            f.write(json.dumps(dict(output)) + '\n')


if __name__ == "__main__":
    fire.Fire(retrieve)
    
    