from PIL import Image
import io
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def bytes_to_tensor(img_bytes: bytes, tfm, device: str):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # (1,C,H,W)
    return x