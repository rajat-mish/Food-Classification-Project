# 🍽️ Food11 Image Classifier Web App

A deep learning-powered web application that classifies food images into 11 categories using a fine-tuned MobileNetV2 model. Built with **TensorFlow** and **Streamlit**, this app offers an interactive way to explore computer vision and food recognition.

---

## 🚀 Live Demo

Upload any food image and get real-time predictions with confidence scores!



---

## 🗂️ Dataset

This project uses the [**Food11 Image Dataset**](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset), containing over 16,000 images categorized into 11 food classes:

- Bread  
- Dairy product  
- Dessert  
- Egg  
- Fried food  
- Meat  
- Noodles-Pasta  
- Rice  
- Seafood  
- Soup  
- Vegetable-Fruit

---

## 🧠 Model Architecture

The model uses **MobileNetV2**, a lightweight convolutional neural network architecture pre-trained on ImageNet and fine-tuned for this task.

### 🔧 Training Pipeline

- **Base Model**: `MobileNetV2` (include_top=False)
- **Custom Layers**:
  - `GlobalAveragePooling2D`
  - `Dense(1024, relu)` + Dropout(0.5)
  - `Dense(512, relu)` + Dropout(0.5)
  - `Dense(11, softmax)`
- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam
- **Callbacks**: ReduceLROnPlateau
- **Trained in Two Phases**:
  1. Feature Extraction (base frozen)
  2. Fine-tuning (base unfrozen with low LR)

---

## 💡 App Features

- Upload food images (JPG/PNG).
- Get predicted food category with confidence.
- Clean and responsive UI.
- Real-time inference using the trained model.

---

