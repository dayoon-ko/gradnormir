import torch 
from tqdm import tqdm
import fire
import os 



def main(model, dataset):
    
    save_pth = f"embeddings/{model}/{dataset}-agg.pt"
    if os.path.exists(save_pth):
        return
    
    cosine = torch.nn.CosineSimilarity(dim=2, eps=1e-08)

    tensors = torch.load(f"embeddings/{model}/{dataset}.pt")
    refs = torch.load(f"embeddings/{model}/nq-train.pt")

    # S_c in Fig 2.
    scores = []
    for i in tqdm(range(tensors.shape[1])):
        tensor = tensors[:, i, :]
        sim = cosine(tensor[:, None, :], refs)
        sim = - torch.topk(sim, 2, dim=-1).values[:, -1]
        scores.append(sim)
    scores = torch.stack(scores)
    #print(scores.shape) # datapoints, # layers

    ref_scores = torch.load(f"embeddings/{model}/nq-train-ref.pt")

    # Aggregation in Fig 3.

    agg_scores = []
    for score in scores:
        sim = cosine(score[None, None, :], ref_scores[:, None, :]).squeeze()
        sim = - torch.topk(sim, 1, dim=-1).values[0]
        agg_scores.append(sim)
        
    agg_scores = torch.stack(agg_scores)
    print("Thrs:", agg_scores.sort().values[int(len(agg_scores)/100 * 95)])
    print(agg_scores.max())
    torch.save(agg_scores, save_pth)

            
            
if __name__ == "__main__":
    fire.Fire(main)