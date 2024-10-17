from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
import os
import torch

def load_vectorstore(
        model_name: str,
        db_faiss_dir: str,
        dropout_rate: float=None,
        pooling: str=None,
        batch_size: int=2048,
        device="cuda"
    ):
    
    # Check directory
    if not os.path.exists(db_faiss_dir):
        raise Exception(f'DB directory {db_root} is invalid.')
    
    # Load embedding object
    embeddings = HuggingFaceEmbeddings(
                    model_name=model_name, 
                    model_kwargs={'device': device},
                    encode_kwargs={
                        'batch_size': batch_size,
                        'device': device}
                )
    
    # Load vectorstore
    db = DocAsQueryFAISS.load_local(
            db_faiss_dir, 
            model_name=model_name, 
            embeddings=embeddings,
            pooling=pooling,
            dropout_rate=dropout_rate
        )
    return db
    

class DocAsQueryFAISS(FAISS):
    
    # Search with document query
    def similarity_search_with_score_dropout(
            self,
            document_query: str,
            k: int,
            **kwargs
        ):
        
        # Apply dropout to the document query
        embedding = self.dropout.last_hidden_state_with_dropout(document_query).tolist()
        
        # Search relevant documents
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            **kwargs,
        )
        return docs

    # Load vectorstores from local
    @classmethod
    def load_local(
        cls, 
        folder_path: str,
        model_name: str,
        embeddings: HuggingFaceEmbeddings,
        pooling: str = None,
        dropout_rate: float = None,
        allow_dangerous_deserialization: bool = True
    ):
        # Load vectorstores
        instance = super().load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        
        if dropout_rate is not None:
            # Register a dropout object
            instance.dropout = EmbeddingModelWithDropout(
                                    model_name=model_name,
                                    dropout_rate=dropout_rate,
                                    pooling=pooling 
                                )
            cls.similarity_search_with_score = cls.similarity_search_with_score_dropout
        return instance
        
        

class EmbeddingModelWithDropout:
    def __init__(self, model_name, dropout_rate, pooling, device="cuda"):
        self.device = device
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        self.load_model(model_name)
        
    def load_model(self, model_name):
        self.model_name = model_name 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model = model.to(self.device)

    def dropout_to_input(self, input_tensor):
        bernoulli_mask = torch.rand_like(input_tensor, device="cuda") > self.dropout_rate
        masked_tensor = input_tensor * bernoulli_mask
        return masked_tensor
    
    def last_hidden_state_with_dropout(self, text):
        # Tokenize input text
        encoded_input = self.tokenizer(
                                text,
                                padding=True,
                                max_length=512,
                                truncation=True,
                                return_tensors="pt",
                            )
        encoded_input = {
                key: val.to(self.device) for key, val in encoded_input.items()
            }
        
        # Get embedding 
        embedding_output = self.model.embeddings(
                                    input_ids=encoded_input["input_ids"],
                                    position_ids=None,
                                    token_type_ids=None,
                                    inputs_embeds=None,
                                    past_key_values_length=0,
                                )
        
        # Get the last hidden state
        prev_output = embedding_output
        for i in range(len(self.model.encoder.layer) - 1):
            encoder_output = self.model.encoder.layer[i](prev_output)[0]
            prev_output = encoder_output
        encoder_output = self.model.encoder.layer[-1](prev_output)[0]
        encoder_output = self.dropout_to_input(encoder_output)
        output_tensor = encoder_output.detach().cpu() # (1, #tokens, hidden_dim)

        # Pooling
        if self.pooling == "cls":
            output_tensor = output_tensor[:, 0, :].squeeze(0) 
        elif self.pooling == "mean":
            output_tensor = output_tensor.mean(dim=1).squeeze(0) # (hidden_dim)
        else:
            raise Exception(f"Pooling method {self.pooling} is not implemented.")
        
        return output_tensor


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
    
