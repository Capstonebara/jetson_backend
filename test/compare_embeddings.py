import json
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import torch

def compare_embeddings(file1_path, file2_path):
    # Load embeddings from JSON files
    with open(file1_path, 'r') as f1:
        embedding1 = json.load(f1)
    
    with open(file2_path, 'r') as f2:
        embedding2 = json.load(f2)
    
    # Convert to numpy arrays
    emb1 = np.array(embedding1[0] if isinstance(embedding1, list) and len(embedding1) == 1 else embedding1)
    emb2 = np.array(embedding2[0] if isinstance(embedding2, list) and len(embedding2) == 1 else embedding2)
    print(f"embed 1{emb1}")
    print(f"embed 2{emb2}")
    
    # Convert numpy arrays to torch tensors
    emb1_tensor = torch.tensor(emb1, dtype=torch.float32)
    emb2_tensor = torch.tensor(emb2, dtype=torch.float32)

    # Calculate cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
    cosine_similarity = 1 - cosine(emb1, emb2)
    
    # Calculate euclidean distance (smaller = more similar)
    euclidean_distance = euclidean(emb1, emb2)

    # Check if tensors are almost equal using PyTorch
    check_similar = torch.allclose(emb1_tensor, emb2_tensor, atol=1e-6)
    
    print(f"Cosine similarity: {cosine_similarity:.6f} (higher is more similar)")
    print(f"Euclidean distance: {euclidean_distance:.6f} (lower is more similar)")
    print(f"Check similar: {check_similar}")
    
    # Typical threshold for face matching
    if cosine_similarity > 0.5:
        print("MATCH: These likely represent the same person")
    else:
        print("NO MATCH: These likely represent different people")

# Usage example
compare_embeddings('server_embed.json', 'face_embedding_new.json')