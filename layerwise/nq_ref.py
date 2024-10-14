import torch 
from tqdm import tqdm
import fire

def main(model = "contriever"):
    cosine = torch.nn.CosineSimilarity(dim=2, eps=1e-08)

    tensors = torch.load(f"embeddings/{model}/nq-train.pt")

    scores = []
    for i in tqdm(range(tensors.shape[1])):
        tensor = tensors[:, i, :]
        sim = cosine(tensor[:, None, :], tensors)
        sim = - torch.topk(sim, 2, dim=-1).values[:, -1]
        scores.append(sim)
    scores = torch.stack(scores)
    print(scores.shape) # datapoints, # layers

    torch.save(scores, f"embeddings/{model}/nq-train-ref.pt")
    
if __name__ == "__main__":
    fire.Fire(main)