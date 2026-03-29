# model/inference.py

import torch
import torchvision.transforms as transforms
from PIL import Image

# TODO: replace with your actual model class
from torchvision import models

# class names (replace with your breeds)
CLASS_NAMES = ["Gir", "Sahiwal", "Jersey"]

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
    model.eval()
    
    return model

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(image_path):
    model = load_model()
    image = preprocess(image_path)
    
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
    
    confidence, pred = torch.max(probs, dim=1)
    
    confidence = confidence.item()
    pred = pred.item()
    
    if confidence < 0.6:
        return {"breed": "unknown", "confidence": confidence}
    
    return {
        "breed": CLASS_NAMES[pred],
        "confidence": confidence
    }