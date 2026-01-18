# src/train_resnet50v2_optimized.py
import os
import copy
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------------------------
# Reproducibility
# -------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------
# Config
# -------------------------
data_dir = "data"
models_root = "models"
save_dir = os.path.join(models_root, "resnet50v2_optimized")
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 25
learning_rate = 3e-5

print("Device:", device)

# -------------------------
# Augmentations
# -------------------------
train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.2)
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# Datasets / Loaders
# -------------------------
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)

# -------------------------
# LOAD RESNET50 V2 (torchvision)
# -------------------------
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze deeper layers only
for name, param in model.named_parameters():
    if any(layer in name for layer in ["layer2", "layer3", "layer4"]):
        param.requires_grad = True

# Replace final FC layer (ResNet50 V2 uses model.fc)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, num_classes)
)
model = model.to(device)

# -------------------------
# Class weights for imbalance
# -------------------------
train_targets = train_dataset.targets
class_counts = [train_targets.count(i) for i in range(num_classes)]
total = sum(class_counts)
weights_list = [total/c for c in class_counts]
class_weights = torch.tensor(weights_list).float().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Optimizer + Scheduler + AMP
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=learning_rate,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# -------------------------
# Training Loop
# -------------------------
best_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())

train_loss_hist, val_loss_hist = [], []
train_acc_hist, val_acc_hist = [], []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # -------- TRAIN --------
    model.train()
    running_loss, running_corrects = 0, 0

    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / len(train_loader.dataset)

    train_loss_hist.append(epoch_loss)
    train_acc_hist.append(epoch_acc)

    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    scheduler.step(epoch)

    # -------- VALIDATE --------
    model.eval()
    val_loss, val_correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="Validation")
        for inputs, labels in pbar_val:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct / len(val_loader.dataset)

    val_loss_hist.append(val_epoch_loss)
    val_acc_hist.append(val_epoch_acc)

    print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    # Best model
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        best_wts = copy.deepcopy(model.state_dict())
        torch.save(best_wts, os.path.join(save_dir, "best_resnet50v2.pth"))
        print("🔥 Saved new BEST model!")

# -------------------------
# Save metrics
# -------------------------
metrics = {
    "train_loss": train_loss_hist,
    "val_loss": val_loss_hist,
    "train_acc": train_acc_hist,
    "val_acc": val_acc_hist,
    "best_val_acc": float(best_acc)
}
with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(save_dir, "val_accuracy.txt"), "w") as f:
    f.write(str(best_acc))

# -------------------------
# Confusion Matrix
# -------------------------
model.load_state_dict(best_wts)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("ResNet50V2 Confusion Matrix")
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()

# -------------------------
# Training curves
# -------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(train_acc_hist, label="Train Acc")
plt.plot(val_acc_hist, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "training_curve.png"))
plt.close()

print("\nTraining complete.")
print("Best Validation Accuracy:", best_acc)
print("All outputs saved in:", save_dir)
