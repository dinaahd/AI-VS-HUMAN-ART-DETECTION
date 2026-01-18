import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ================================
# ⚙️ Config
# ================================
data_dir = "data/val"  # Folder containing validation/test images
model_path = "models/resnet_finetuned.pth"
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 🧩 Data Preparation
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
class_names = dataset.classes

# ================================
# 🧠 Load Model
# ================================
model = models.resnet50(weights='IMAGENET1K_V2')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print(f"✅ Loaded model: {model_path}")
print(f"✅ Classes: {class_names}")

# ================================
# 📊 Evaluation
# ================================
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ================================
# 📈 Metrics
# ================================
cm = confusion_matrix(all_labels, all_preds)
acc = np.mean(np.array(all_preds) == np.array(all_labels))
report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)

print("\n🎯 Accuracy:", round(acc * 100, 2), "%")
print("\n📋 Classification Report:\n", report)

# ================================
# 🎨 Visualization
# ================================
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="mako", 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - AI Art Detection", fontsize=14, weight='bold', color='#00FFFF')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()

print("\n🖼️ Confusion matrix saved as models/confusion_matrix.png")
