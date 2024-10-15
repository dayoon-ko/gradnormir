from transformers import AutoModel
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
import logging
from tqdm import tqdm
from glob import glob

def get_logger(log_path):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger    
    
def store_data(
        data_root: str,
        dataset_name: str,
        glob_dir: str,
        db_faiss_dir: str,
        batch_size: int,
        model_name: str,
        logging_path: str = None, 
        use_metadata: bool = False, 
        max_length: str = 512,
        dimension: int = None,
        device: str = "cuda"
    ):
    
    # Define the metadata extraction function.
    def metadata_func(record: dict, metadata: dict) -> dict:
        del record["text"]
        metadata.update(record)
        return metadata
    
    # Get logger
    logging_path = logging_path if logging_path is not None else f"{db_faiss_dir}/index.log"
    logger = get_logger(logging_path)
    
    # Check whether already created
    if os.path.exists(f"{db_faiss_dir}/index.faiss"):
        logger.info(f"Already created in {db_faiss_dir}")
    elif not os.path.exists(db_faiss_dir):
        os.makedirs(db_faiss_dir, exist_ok=True)
    
    # Set parameters
    dataset_path = f"{data_root}/{dataset_name}/{glob_dir}"
    jq_schema = "."
    content_key = "text"
    json_lines = True
    
    # Load jsonl files
    loader = JSONLoader(
                dataset_path,
                jq_schema=jq_schema,  
                content_key=content_key,
                json_lines=json_lines,
                metadata_func=metadata_func
            )
    documents = loader.load()
    logger.info(f"Documents are loaded from {dataset_path}")
    logger.info(f'# Documents: {len(documents)}')
    
    # Load embedding object
    embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': device},
                    encode_kwargs={'batch_size': batch_size, 
                                   'max_length': max_length,
                                   'device': device}
                )
    
    # Generate a database
    if dimension is None:
        model = AutoModel.from_pretrained(model_name)
        dimension = model.config.hidden_size
    index = faiss.IndexFlatIP(dimension) 
    vector_store = FAISS(
        embedding_function=embeddings, 
        index=index, 
        docstore= InMemoryDocstore(), 
        index_to_docstore_id={} 
    )
    
    # Save documents
    logger.info(f'Saving embeddings to {db_faiss_dir}')
    vector_store.add_documents(documents=documents)
    vector_store.save_local(f'{db_faiss_dir}')
    logger.info('Saved')
        
        
    
if __name__ == "__main__":
    fire.Fire(store_data)