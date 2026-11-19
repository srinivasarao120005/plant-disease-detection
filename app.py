import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("Plant Disease Detection")
mp=Path("models/plant_disease.keras"); cp=Path("models/class_names.json")
if not mp.exists() or not cp.exists():
    st.warning("Train the model first: python train.py"); st.stop()
model=tf.keras.models.load_model(mp); names=json.loads(cp.read_text())
file=st.file_uploader("Upload a leaf image",type=["jpg","jpeg","png"])
if file:
    image=Image.open(file).convert("RGB").resize((224,224))
    st.image(image,caption="Uploaded leaf",width=320)
    arr=np.expand_dims(np.asarray(image,dtype=np.float32),0)
    probs=model.predict(arr,verbose=0)[0]
    idx=int(np.argmax(probs))
    st.subheader(names[idx].replace("___"," — "))
    st.metric("Model confidence",f"{probs[idx]*100:.1f}%")
    st.caption("Prediction is limited to the model's trained classes; confirm field diagnosis with an expert.")
