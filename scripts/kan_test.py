import torch
import torchvision
import cv2
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

model = torch.jit.load("./saved_model/kan_model.pt")
model.eval()
model = model.to(device)

print(next(model.parameters()).device)

image = cv2.imread("./1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])

image_tensor = transform(image).unsqueeze(0).to(device)
print(image_tensor.device)
print(model(image_tensor))
