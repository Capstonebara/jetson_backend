from embedding_model.EdgeFaceKan import EdgeFaceKAN
import torch
from torchvision import transforms
from PIL import Image
import json

embedding_model = EdgeFaceKAN(num_features=512, grid_size=15, rank_ratio=0.5, neuron_fun="mean")
checkpoint = torch.load("./model.pt", map_location=torch.device("cpu"))
embedding_model.load_state_dict(checkpoint)
embedding_model.eval()
transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

image = Image.open("./Straight-1744807079700.jpg")
face_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    embedding = embedding_model(face_tensor)
    print(embedding.shape)
    print(embedding)

    # Convert embedding to a serializable format
    embedding_np = embedding.cpu().numpy().tolist()
    
    # Save to JSON file
    output_filename = "face_embedding_new.json"
    with open(output_filename, 'w') as f:
        json.dump(embedding_np, f)
        
    print(f"Embedding saved to {output_filename}")
