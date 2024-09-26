from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore 

import os
import fire
import torch
import faiss 
from tqdm import tqdm
from glob import glob

import warnings
from langchain_core.globals import set_verbose, set_debug

# Ignore all warnings
warnings.filterwarnings("ignore")

set_verbose(False)
set_debug(False)

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
        max_length: str = 512,
        dimension: int = 1024
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
                        'max_length': max_length,
                        'device': 'cuda',
                    }
                )    
    # Make a DB
    if db_faiss_dir is None:
        db_faiss_dir = f"vectorstore/{dataset_name}"
    print(f'Extract db from documents {db_faiss_dir}')
    
    '''
    db = FAISS.from_documents(
            documents, 
            embeddings,
            normalize_L2 = True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
    '''
    
    index = faiss.IndexFlatIP(dimension) 
    vector_store = FAISS(
        embedding_function=embeddings, 
        index=index, 
        docstore= InMemoryDocstore(), 
        index_to_docstore_id={} 
    )
    vector_store.add_documents(documents=documents)

    print(f'Saving embeddings to {db_faiss_dir}')
    vector_store.save_local(f'{db_faiss_dir}')
    print('Saved')
        
        
    
if __name__ == "__main__":
    fire.Fire(store_data)