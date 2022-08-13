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
tab1, tab2 = st.tabs(["About", "Test"])
with tab1:
    st.header("ResNet On ImageNet")
    st.markdown("PyTorch, specifically, torchvision has a lot of models and pre-trained weights. CNN Models are mostly trained on ImageNet. Hyperparameters can be found [here](https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621).")
with tab2:
    st.header("Demo Classification")
    image = st.file_uploader(label="")
    if image is not None:
        img = Image.open(image)
        st.image(image, caption='Uploaded Image')
        if st.button("Classify"):
            name, score = classify(img)        
            st.write(f"Predicted to be {name} with certainty {score}%.")