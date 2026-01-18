import torch
from torchvision import models, transforms
from PIL import Image
import os

# ===============================
# 🔧 Configuration
# ===============================
model_path = 'models/resnet_finetuned.pth'
class_names = ['ai_art', 'human_art']  # same order as your training set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
# 🧠 Load Model
# ===============================
model = models.resnet50(weights='IMAGENET1K_V2')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ===============================
# 🖼️ Preprocessing
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# 🔍 Prediction Function
# ===============================
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds].item()

    label = class_names[preds[0]]
    print(f"🖼️ Image: {os.path.basename(image_path)}")
    print(f"🎨 Predicted: {label} ({confidence*100:.2f}% confidence)")

# ===============================
# 🚀 Test with any image
# ===============================
if __name__ == "__main__":
    test_path = input("Enter image path: ")
    if os.path.exists(test_path):
        predict_image(test_path)
    else:
        print("❌ File not found!")
