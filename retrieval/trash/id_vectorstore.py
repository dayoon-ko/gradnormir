from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy

import os
import fire
import torch
from tqdm import tqdm
from glob import glob


# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:
    
    del record["text"]
    metadata.update(record)
    
    return metadata

def store_data(
        dataset_name = "trec-covid",
        glob_dir:str = "corpus.jsonl",
        data_dir: str = '/gallery_louvre/dayoon.ko/research/sds/src/datasets',
        db_faiss_dir: str = None,
        batch_size: int = 64,
        model_name: str = "intfloat/multilingual-e5-large", #'sentence-transformers/all-MiniLM-L6-v2'
        use_metadata: bool = False, 
    ):
    
    # Document
    loader = JSONLoader(
                f"{data_dir}/{dataset_name}/{glob_dir}", 
                jq_schema=".",  
                content_key="text",
                json_lines=True,
                metadata_func=metadata_func
            )
    documents = loader.load()
    print(f'Document count: {len(documents)}')
    
    # Split document
    embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        'device': 'cuda',
                    },
                    encode_kwargs={
                        'batch_size': batch_size,
                        'device': 'cuda',
                    }
                )    
    # Make a DB
    if db_faiss_dir is None:
        db_faiss_dir = dataset_name
    print(f'Extract db from documents {db_faiss_dir}')
    db = FAISS.from_documents(
            documents, 
            embeddings,
            normalize_L2 = True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
    print(f'Saving embeddings to {db_faiss_dir}')
    db.save_local(f'{db_faiss_dir}')
    print('Saved')
        
        
    
if __name__ == "__main__":
    fire.Fire(store_data)