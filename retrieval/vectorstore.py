from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        batch_size: int = 256,
        model_name: str = "intfloat/multilingual-e5-large", #'sentence-transformers/all-MiniLM-L6-v2'
        use_metadata: bool = False, 
    ):
    
    # Document
    #loader = DirectoryLoader(data_dir, glob=glob_dir, show_progress=True, loader_cls=JSONLoader, loader_kwargs={'jq_schema':'.text', 'json_lines':True})
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
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    #splits = text_splitter.split_documents(documents)
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
    db = FAISS.from_documents(documents, embeddings)
    print(f'Saving embeddings to {db_faiss_dir}')
    db.save_local(f'{db_faiss_dir}')
    print('Saved')
        
        
    
if __name__ == "__main__":
    fire.Fire(store_data)