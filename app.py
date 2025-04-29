import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/food_model.h5')

model = load_model()

# Class labels
class_labels = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
                'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']

# App Title
st.title("🍽️ Food Image Classifier - Food11 Dataset")

# Upload Image
uploaded_file = st.file_uploader("Upload a food image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    st.markdown(f"### 🍱 Predicted: **{class_labels[predicted_class]}**")
    st.markdown(f"Confidence: **{confidence*100:.2f}%**")
