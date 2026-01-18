# src/train_resnet50v2_ultimate.py
"""
Ultimate ResNet50-V2 training script (AI-vs-Human art).
SAFE FOR WINDOWS MULTIPROCESSING.

Saves inside: models/resnet50v2_optimized/

Outputs:
 - best_model.pth
 - training_metrics.json
 - val_accuracy.txt
 - test_accuracy.txt
 - confusion_matrix.png
 - training_curve.png
"""

import os
import copy
import json
import random
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================
# CONFIG
# ============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================
# Focal Loss
# ============================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ============================
# MAIN TRAINING FUNCTION
# ============================
def main():

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------- Paths --------
    data_dir = "data"
    save_dir = os.path.join("models", "resnet50v2_optimized_ultimate")
    os.makedirs(save_dir, exist_ok=True)

    # -------- Hyperparams --------
    NUM_CLASSES = 2
    BATCH_SIZE = 24
    NUM_EPOCHS = 30
    BASE_LR = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4   # 🔥 Now safe on Windows
    GRAD_ACCUM_STEPS = 1

    # -------- Transforms --------
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(0.1, 0.1, 0.05, 0.01),
        transforms.RandomAdjustSharpness(1.5, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # -------- Datasets --------
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"), val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    class_names = train_ds.classes
    print("Classes:", class_names)

    # -------- Class Weights --------
    counts = [train_ds.targets.count(i) for i in range(NUM_CLASSES)]
    total = sum(counts)
    weights = [total / c for c in counts]
    class_weights = torch.tensor(weights).float().to(device)

    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    focal = FocalLoss(gamma=2.0, weight=class_weights)

    def loss_fn(logits, labels):
        return 0.6 * ce(logits, labels) + 0.4 * focal(logits, labels)

    # -------- Model --------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features

    # MATCHES STREAMLIT STRUCTURE
    model.fc = nn.Sequential(
        nn.Identity(),
        nn.Linear(in_features, NUM_CLASSES)
    )

    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = True

    # -------- Optimizer + Scheduler --------
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    def warmup_lambda(epoch):
        if epoch < 3:
            return (epoch + 1) / 3
        return 1

    scheduler_warm = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    scheduler_cos  = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # -------- Training Loop --------
    best_val = 0
    no_improve = 0
    patience = 6

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n🚀 Training Started...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # ---- Training ----
        model.train()
        total_loss = 0
        total_correct = 0

        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(inputs)
                loss = loss_fn(logits, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            preds = torch.argmax(logits, 1)
            total_loss += loss.item() * inputs.size(0) * GRAD_ACCUM_STEPS
            total_correct += (preds == labels).sum().item()

        train_loss = total_loss / len(train_ds)
        train_acc = total_correct / len(train_ds)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        if epoch < 3:
            scheduler_warm.step()
        else:
            scheduler_cos.step(epoch)

        # ---- Validation ----
        model.eval()
        v_loss = 0
        v_correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = loss_fn(logits, labels)

                preds = torch.argmax(logits, 1)
                v_loss += loss.item() * inputs.size(0)
                v_correct += (preds == labels).sum().item()

        val_loss = v_loss / len(val_ds)
        val_acc = v_correct / len(val_ds)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ---- Save Best ----
        if val_acc > best_val:
            best_val = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, os.path.join(save_dir, "best_model.pth"))
            print(f"🔥 SAVED NEW BEST MODEL (Acc={best_val:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("⏳ Early stopping activated.")
            break

    # ================================
    # SAVE METRICS
    # ================================
    torch.save(best_wts, os.path.join(save_dir, "best_model.pth"))

    with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
        json.dump(hist, f, indent=4)

    with open(os.path.join(save_dir, "val_accuracy.txt"), "w") as f:
        f.write(str(best_val))

    # ================================
    # TEST EVALUATION
    # ================================
    model.load_state_dict(best_wts)
    model.eval()

    correct = 0
    total = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    test_acc = correct / total
    print(f"\n🎯 TEST ACCURACY: {test_acc:.4f}")

    with open(os.path.join(save_dir, "test_accuracy.txt"), "w") as f:
        f.write(str(test_acc))

    # ================================
    # CONFUSION MATRIX
    # ================================
    cm = confusion_matrix(labels_all, preds_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - ResNet50v2")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # ================================
    # TRAIN CURVES
    # ================================
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["train_loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(hist["train_acc"], label="Train Acc")
    plt.plot(hist["val_acc"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    plt.close()

    print(f"\n✨ ALL RESULTS SAVED TO: {save_dir}")


# ============================
# WINDOWS-SAFE ENTRY POINT
# ============================
if __name__ == "__main__":
    main()
