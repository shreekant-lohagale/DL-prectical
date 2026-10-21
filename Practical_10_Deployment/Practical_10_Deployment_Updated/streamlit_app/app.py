import numpy as np
import streamlit as st
from model_utils import load_mnist_model

st.title("MNIST Digit Prediction App")
st.caption("Practical 10: pixel-value demonstration from DL-Practicals.docx")

@st.cache_resource
def get_model():
    return load_mnist_model()

model = get_model()
value = st.slider("Enter Pixel Value", 0, 255, 100)
if st.button("Predict"):
    sample = np.full((1, 28, 28), value / 255.0, dtype=np.float32)
    probabilities = model.predict(sample, verbose=0)
    result = int(np.argmax(probabilities, axis=1)[0])
    st.success(f"Predicted Digit: {result}")
    st.caption("A constant-intensity image is a deployment demo, not a handwritten digit. Predictions may not be meaningful.")
