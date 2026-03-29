import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from pathlib import Path

# PATHS - Use absolute paths or relative from script location
script_dir = Path(__file__).parent.parent
DATA_DIR = str(script_dir / "data" / "train")
VAL_DIR = str(script_dir / "data" / "val")

# CONFIG
BATCH_SIZE = 8
EPOCHS = 5
LR = 0.001

# TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

# DATASET
print(f"Loading from: {DATA_DIR}")
print(f"Validation from: {VAL_DIR}")
print(f"Training dir exists: {os.path.exists(DATA_DIR)}")
print(f"Validation dir exists: {os.path.exists(VAL_DIR)}")

try:
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

print(f"\n✅ Train samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"✅ Classes: {train_dataset.classes}")

if len(train_dataset) == 0:
    print("⚠️  WARNING: No training samples found! Check your data directory.")
    exit(1)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# MODEL
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    print(f"\nEpoch {epoch+1} started")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

    # VALIDATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# SAVE MODEL
model_path = str(script_dir / "model" / "model.pth")
classes_path = str(script_dir / "model" / "classes.txt")

torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# SAVE CLASS NAMES
with open(classes_path, "w") as f:
    for cls in train_dataset.classes:
        f.write(cls + "\n")
print(f"Classes saved to: {classes_path}")

print("\n✅ Training complete. Model saved.")