# 🖼️ AI vs Human Art Detection
## Deep Learning Classification using ResNet50V2
## 📌 Overview

This project is an AI-powered image classification system that accurately distinguishes between AI-generated artwork and human-created artwork using a fine-tuned ResNet50V2 deep learning model.

With the rapid rise of generative AI, identifying synthetic art has become critical for digital authenticity, copyright protection, and ethical AI use. This project addresses that challenge using state-of-the-art computer vision techniques.
---

## 🚀 Key Features

- ✅ Binary classification: AI Art vs Human Art

- ✅ Transfer learning with ResNet50V2

- ✅ Optimized training pipeline

- ✅ Detailed evaluation metrics & visualizations

- ✅ Modular, clean, and scalable codebase

-✅ Ready for deployment & future extension
---

## 🧠 Model Architecture

Backbone: ResNet50V2 (pretrained on ImageNet)

Technique: Transfer Learning + Fine-tuning

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Evaluation Metrics:

Accuracy

Confusion Matrix

Training vs Validation Curves
---

## 📂 Project Structure
AI-VS-HUMAN-ART-DETECTION/
│
├── app/
│   └── final.py                  # Application / inference entry point
│
├── assets/
│   ├── confusion_matrix.png      # Final evaluation visualization
│   └── training_curve.png        # Training vs validation curves
│
├── data/
│   ├── train/                    # Training images
│   ├── val/                      # Validation images
│   └── test/                     # Test images
│
├── models/
│   └── resnet50v2/
│       ├── confusion_matrix.png
│       ├── training_curve.png
│       ├── training_metrics.json
│       └── val_accuracy.txt
│
├── src/
│   ├── dataset_prep.py           # Dataset preprocessing
│   ├── train_resnet50v2_optimized.py
│   ├── train_resnet50v2_ultimate.py
│   ├── evaluate.py               # Model evaluation
│   ├── test_evaluate_resnet50v2.py
│   └── predict.py                # Prediction script
│
├── .gitignore
└── README.md
---

## 📊 Results

Strong validation accuracy on unseen data

Clear separation between AI-generated and human-made images

Confusion matrix and learning curves demonstrate stable training and low overfitting

(See assets/ and models/resnet50v2/ for visual results)
---

## ⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/AI-VS-HUMAN-ART-DETECTION.git
cd AI-VS-HUMAN-ART-DETECTION

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt


(If requirements.txt is not present, install manually: TensorFlow, NumPy, OpenCV, Matplotlib, Scikit-learn)

## 🏋️ Training the Model
python src/train_resnet50v2_ultimate.py


or (optimized version):

python src/train_resnet50v2_optimized.py

## 🧪 Evaluating the Model
python src/evaluate.py


or:

python src/test_evaluate_resnet50v2.py

## 🔮 Making Predictions
python src/predict.py --image path/to/image.jpg

## 🌍 Use Cases

🎨 Digital art authentication

🛡️ AI-generated content detection

📰 Media & journalism verification

🧠 AI ethics and research

🏆 Hackathons & academic projects
---

## 🔮 Future Improvements

🔹 Web app deployment (Streamlit / FastAPI)

🔹 Support for multi-class detection

🔹 Explainable AI (Grad-CAM visualization)

🔹 Larger and more diverse datasets

🔹 Model benchmarking with ViT & EfficientNet
---

## 🧑‍💻 Author

Dina
Computer Science Undergraduate | AI & ML Enthusiast
📍 India

🔗 GitHub: https://github.com/dinaahd
🔗 LinkedIn: https://www.linkedin.com/in/dina-ahd
---

## ⭐ Acknowledgements

ResNet architecture by Microsoft Research

ImageNet pretrained weights

Open-source AI & ML community
---

## 📜 License

This project is licensed for academic and educational use.
---