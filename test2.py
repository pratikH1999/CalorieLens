import tensorflow as tf
import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import google.generativeai as genai
import os

# Load CNN model for food classification
cnn_model = load_model('FV.h5')

# Class labels
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Gemini API configuration
genai.configure(api_key="AIzaSyBV0liOMF_rR0h3MN3KXKC1pdtVuE7kgvY")

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Preprocessing for MiDaS
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])


def prepare_image(img_path):
    """
    Prepares the image for food classification and depth estimation.
    Returns the classified food item and estimated depth value.
    """
    # Load and preprocess image for classification
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict class
    answer = cnn_model.predict(img)
    y_class = answer.argmax(axis=-1)[0]
    res = labels[y_class].capitalize()

    # Load image for depth estimation
    img_cv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_depth = cv2.resize(img_rgb, (384, 384))
    img_depth = transform(img_depth).unsqueeze(0)

    # Predict depth
    with torch.no_grad():
        depth_map = midas(img_depth)
        depth_value = depth_map.mean().item()  # Extract average depth value

    print(f"Predicted: {res}, Estimated Depth: {depth_value}")
    return res, depth_value


def load_image(img_path):
    """Loads an image using PIL."""
    return Image.open(img_path)


def calculate_calories_gemini(res, depth_value):
    """
    Estimates food volume based on depth and uses Gemini API to return ONLY the calorie value.
    """
    # Approximate volume based on depth value
    estimated_volume = round((depth_value * 5), 2)  # Adjustable scaling factor

    prompt = f"""
    You are an expert in nutrition. Given the food item '{res}' with an estimated volume of {estimated_volume} cm³,
    return ONLY the approximate calorie content as a single number. Do not include any other text.
    """

    # Call Gemini API
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt])

    # Extract only the numeric value from the response (if it contains extra text)
    try:
        calories = float(response.text.strip().split()[0])  # Extract first number
    except ValueError:
        calories = "Error: Invalid response from Gemini"

    return calories


def simple(result, depth_value):
    """Displays the classification result and estimated calories using Gemini API."""
    category = "**Category : Vegetables**" if result in vegetables else "**Category : Fruit**"
    st.info(category)
    st.success(f"**Predicted : {result}**")

    # Use Gemini API for calorie estimation
    calories = calculate_calories_gemini(result, depth_value)

    # Display only the calorie number
    #st.info("Calorie Estimation:")
    st.info(f"**Calories Estimation : {calories} kcal**")


def run():
    """Streamlit UI to classify food and estimate calories."""
    st.title("📷 Calories Lens 🍽️")
    st.subheader("Snap a photo & get instant calorie insights!")

    img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)

        # Ensure upload directory exists
        save_dir = "./upload_images/"
        os.makedirs(save_dir, exist_ok=True)
        save_image_path = os.path.join(save_dir, img_file.name)

        # Save uploaded file
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Get classification and depth estimation
        result, depth_value = prepare_image(save_image_path)

        # Display classification and calorie estimation
        simple(result, depth_value)

        # Display depth estimation result (as text, since it's a numerical value)
        st.subheader("Depth Estimation Result")
        st.info(f"Estimated Depth Value : {depth_value:.2f}")

run()
