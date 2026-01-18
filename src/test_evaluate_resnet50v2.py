# src/test_evaluate_resnet50v2.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------
# Paths
# ----------------------------
test_data_dir = "data/test"

# Folder format identical to baseline models:
model_dir = "models/resnet50v2_optimized"
model_path = os.path.join(model_dir, "best_resnet50v2.pth")

save_dir = os.path.join(model_dir, "test_results")
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# Transforms
# ----------------------------
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ----------------------------
# Dataset + Loader
# ----------------------------
test_dataset = datasets.ImageFolder(test_data_dir, transform=test_tf)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)
print("Test Classes:", class_names)

# ----------------------------
# Load Model (Torchvision ResNet50-V2)
# ----------------------------
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

# Replace classifier to match training
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, num_classes)
)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Evaluation
# ----------------------------
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = correct / total
print(f"\n🎯 TEST ACCURACY (ResNet50V2 Optimized): {test_accuracy:.4f}")

# Save test accuracy
with open(os.path.join(save_dir, "test_accuracy.txt"), "w") as f:
    f.write(str(test_accuracy))

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(cmap=plt.cm.Blues)
plt.title("Test Confusion Matrix - ResNet50V2 Optimized")
plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
plt.close()

print(f"\n📁 Test evaluation saved in: {save_dir}")
