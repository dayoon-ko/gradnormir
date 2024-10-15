from torch.utils.data import Dataset, DataLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import os
import fire
import json
from utils import Retrieval


def load_vectorstore(
        model_name: str,
        db_faiss_dir: str,
        batch_size: int=2048,
        device="cuda"
    ):
    
    # if vectorstore exists
    if not os.path.exists(f'{db_faiss_dir}/index.faiss'):
        raise Exception(f'DB directory {db_root} is invalid.')
    embeddings = HuggingFaceEmbeddings(
                    model_name=model_name, 
                    model_kwargs={'device': device},
                    encode_kwargs={'batch_size':batch_size, 'device': device}
                )
    db = FAISS.load_local(
            db_faiss_dir, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        ) 
    return db
    

class RetrievalDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
    ):  
        # read data
        self.data_path = data_path
        with open(self.data_path) as f:
            self.dataset = [json.loads(i) for i in f.readlines()]
        print(f'Total {len(self.dataset)} data points') 
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
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
        top_k: int = 100,
    ):
    
    # Load dataset
    data_path = f"{data_root}/{dataset_name}/queries.jsonl"
    print("Data path:", data_path)
    dataset = RetrievalDataset(data_path)    
    
    # Get dataloader
    dataloader = DataLoader(
                    dataset, 
                    batch_size=1, 
                    collate_fn=dataset.collate_fn
                )
    
    # Make a retrieval chain
    print(f"Load db from {db_faiss_dir}")
    retriever_db = load_vectorstore(
                        model_name=model_name,
                        db_faiss_dir=db_faiss_dir
                    )
    retrieval = Retrieval(retriever_db, search_kwargs={"k":top_k})
    
    # Path to save
    save_path = f'{save_root}/q2d-retrieval.jsonl'
    print(f"Save results to {save_path}")
        
    # execute inference
    for _, meta in tqdm(enumerate(dataloader), total=len(dataloader)):
        retrieved_doc_ids = retrieval(meta["text"]) 
        with open(save_path, 'a') as f:
            output = {"_id": meta["_id"], "retrieval": retrieved_doc_ids}
            f.write(json.dumps(output) + '\n')


if __name__ == "__main__":
    fire.Fire(retrieve)
    
    