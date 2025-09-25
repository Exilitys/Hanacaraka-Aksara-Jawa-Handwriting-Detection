import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import pickle as p

# --- Config ---
IMG_SIZE = (224, 224)
MODEL_PATH = "model\\ResNet50-FineTune-40Fl.keras"  # change to your model path

# --- Load Model ---
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

file = open('model\\class-indicies.pkl', 'rb')
class_indicies = p.load(file)

label_map = {v: k for k, v in class_indicies}

# --- Streamlit App ---
st.title("Image Classification with Keras Model")
st.write("Upload an image to classify it using a trained model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width =True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred_probs = model.predict(img_array)[0]
    pred = np.argmax(pred_probs)
    label = label_map[pred]

    st.subheader("Prediction Results")
    st.title(label)