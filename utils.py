import os
import json
import cv2
import faiss
import numpy as np
from PIL import Image

def load_embedding_from_json(embedded_folder: str):
    embeddings = {}
    for subfolder in os.listdir(embedded_folder):
        subfolder_path = os.path.join(embedded_folder, subfolder)
        # print(subfolder_path)
        label = subfolder_path.split(" ")[1]
        # print(label)
        if os.path.isdir(subfolder_path):
            embeddings[label] = []
            for json_file in os.listdir(subfolder_path):
                if json_file.endswith('.json'):
                    # print(json_file)
                    json_path = os.path.join(subfolder_path, json_file)
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        # print(data)
                        embeddings[label].append(data)
    return embeddings


def build_faiss_index(embeddings, embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)
    label_map = {}
    
    for label, embs in embeddings.items():
        start = index.ntotal
        
        xb = np.array(embs).astype('float32')
        index.add(xb)
        
        end = index.ntotal
        label_map[label] = (start, end - 1)
        print(f"Added {len(embs)} embeddings for '{label}' "
            f"into positions {start} to {end-1}.")
    
    return index, label_map

