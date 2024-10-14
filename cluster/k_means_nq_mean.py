from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer
import torch
import fire

# Clustering algorithm (K-means) and product quantization
class ProductQuantization:
    def __init__(self, num_groups, num_centroids):
        self.num_groups = num_groups
        self.num_centroids = num_centroids
        self.codebooks = []

    def fit(self, document_vectors):
        D = document_vectors.shape[1]
        group_size = D // self.num_groups
        self.codebooks = []

        for i in range(self.num_groups):
            # Slice the vectors into sub-vectors
            sub_vectors = document_vectors[:, i * group_size: (i + 1) * group_size]
            # Normalize sub-vectors for consistency
            sub_vectors = normalize(sub_vectors)
            # Apply K-means clustering to each group
            kmeans = KMeans(n_clusters=self.num_centroids, random_state=42, n_init="auto")
            kmeans.fit(sub_vectors)
            self.codebooks.append(kmeans.cluster_centers_)

    def quantize(self, document_vectors):
        D = document_vectors.shape[1]
        group_size = D // self.num_groups
        quantized_codes = []

        for i in range(self.num_groups):
            sub_vectors = document_vectors[:, i * group_size: (i + 1) * group_size]
            sub_vectors = normalize(sub_vectors)
            centroids = self.codebooks[i]

            # Find the closest centroids for each sub-vector
            distances = np.linalg.norm(sub_vectors[:, None, :] - centroids[None, :, :], axis=2)
            closest_centroids = np.argmin(distances, axis=1)
            quantized_codes.append(closest_centroids)

        # Transpose to get the final docids as a combination of all group assignments
        return np.array(quantized_codes).T
    
    def compute_euclidean_distances(self, sub_vector, centroids):
        # Compute the Euclidean distance between a sub-vector and all centroids in a sub-codebook
        distances = np.linalg.norm(sub_vector - centroids, axis=1)
        return distances

    def quantize_new_document(self, document_vector):
        D = document_vector.shape[0]
        num_groups = len(self.codebooks)
        group_size = D // num_groups
        quantized_codes = []
        distances_all = []
        
        for i in range(num_groups):
            # Extract the sub-vector for the i-th group
            sub_vector = document_vector[i * group_size: (i + 1) * group_size]
            sub_vector = normalize(sub_vector.reshape(1, -1))[0]  # Normalize for consistency
            centroids = self.codebooks[i]

            # Compute distances between the sub-vector and all centroids
            distances = self.compute_euclidean_distances(sub_vector, centroids)
            distances_all.append(np.mean(distances))

            # Find the nearest centroid (minimum distance)
            nearest_centroid_idx = np.argmin(distances)
            quantized_codes.append(nearest_centroid_idx)

        return quantized_codes, distances_all


def main(
    model
):
    # Load ID docs to init clusters
    id_dataset = "nq-train"
    id_embeddings = torch.load(f"/gallery_louvre/dayoon.ko/research/eval_retrieval/layerwise/embeddings/{model}/{id_dataset}.pt",
                            weights_only=True)[-1] # Take the last hidden state

    # Apply Product Quantization to make codebooks
    pq = ProductQuantization(num_groups=8, num_centroids=16)
    pq.fit(id_embeddings) # (#datapoints, #num_groups)

    # Load new document embeddings
    new_embeddings = torch.load(f"/gallery_louvre/dayoon.ko/research/eval_retrieval/layerwise/embeddings/{model}/{id_dataset}.pt",
                            weights_only=True)[-1] # Take the last hidden state

    # Iterate over to get distances
    avg_distance_record = []
    for embedding in tqdm(new_embeddings):
        _, distances = pq.quantize_new_document(embedding)
        avg_distance = sum(distances) / len(distances)
        avg_distance_record.append(avg_distance)
    print(sum(avg_distance_record) / len(avg_distance_record))

if __name__ == "__main__":
    fire.Fire(main)
    
# bge - 0.7559755225570081
# contriever - 0.8858456168888794
# gte - 0.5394490958915853
# me5 - 0.5075114514537833
