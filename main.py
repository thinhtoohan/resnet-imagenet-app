import streamlit as st
import requests
from PIL import Image

from torchvision.models import resnet50, ResNet50_Weights

def classify(image):
    # Step 1: Initialize model with the best available weights
    # image = Image.open(image)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(image).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    score = f"{100*score:.1f}"
    return category_name,score

st.title("ResNet On ImageNet")
image = st.file_uploader("Choose an image")
if image is not None:
    img = Image.open(image)
    st.image(image, caption='Uploaded Image')
if st.button("Classify"):
    img = Image.open(image)
    name, score = classify(img)        
    st.write(f"Predicted to be {name} with certainty {score}%.")