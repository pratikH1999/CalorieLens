import tensorflow as tf
import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

from tensorflow.keras.models import load_model
cnn_model = load_model('FV.h5')
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

input_prompt="""
You are an expert in nutritionist where you need to see the food items from the image
               and calculate the total calories, also provide the details of every food items with calories intake
               is below format

               1. Item 1 - no of calories
               2. Item 2 - no of calories
               ----
               ----


"""

genai.configure(api_key="AIzaSyBV0liOMF_rR0h3MN3KXKC1pdtVuE7kgvY")


def fetch_calories(prediction):
    try:
        url = "https://www.google.com/search?&q=calories+in" + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        #st.error("Can't able to fetch the Calories")
        st.session_state["fetch_failed"] = True  # Mark calorie fetching as failed
        return None
        print(e)


def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = cnn_model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()

def load_image(img_path):
    img = Image.open(img_path) 
    return img

def simple(result):
    if result in vegetables:
        st.info('**Category : Vegetables**')
    else:
        st.info('**Category : Fruit**')
    st.success("**Predicted : " + result + '**')
    cal = fetch_calories(result)
    if cal:
        st.warning('**' + cal + '(100 grams)**')

def gemini(image,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([image,prompt])
    return response.text
    

input_prompt="""
You are an expert in nutritionist where you need to see the food items from the image
               and calculate the total calories, also provide the details of every food items with calories intake
               is below format

               1. Item 1 - no of calories
               2. Item 2 - no of calories
               ----
               ----

and dont any give consideration pls.
"""
def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = '.upload_images' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

    #if st.button("Predict"):
    if img_file is not None:
        result = prepare_image(save_image_path)
        result_image=load_image(save_image_path)
       #response=gemini(result_image,input_prompt)
        #st.subheader("The Response is")
        #st.write(response)
        cal_fetch_failed = False
        try:
            simple(result)
        except Exception:
            cal_fetch_failed = True  # If simple() fails, use Gemini instead

        # If fetching calories failed, use Gemini
        if cal_fetch_failed or st.session_state.get("fetch_failed", False):
            response = gemini(result_image, input_prompt)
            st.subheader("Calorie Estimation:")
            st.write(response)

run()