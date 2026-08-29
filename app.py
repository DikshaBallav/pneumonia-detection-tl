import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)


# ---------------------------------
# Create the same model architecture
# ---------------------------------

@st.cache_resource
def load_model():

    base = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze VGG16 layers
    for layer in base.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Load your trained weights
    model.load_weights("pneumonia_vgg16.weights.h5")

    return model


model = load_model()


# ---------------------------------
# Application
# ---------------------------------

st.title("🫁 Pneumonia Detection")

st.write(
    "Upload a chest X-ray image to classify it as "
    "Normal or Pneumonia."
)


# ---------------------------------
# Upload image
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------
# Prediction
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        # Resize exactly as in your notebook
        image_resized = image.resize((224, 224))

        # Convert to NumPy array
        image_array = np.array(image_resized)

        # Normalize exactly as in your notebook
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )[0][0]

        # Your class mapping:
        # NORMAL = 0
        # PNEUMONIA = 1

        if prediction >= 0.5:

            st.error("⚠️ Pneumonia Detected")

            st.write(
                f"Pneumonia probability: {prediction:.2%}"
            )

        else:

            st.success("✅ Normal")

            st.write(
                f"Normal probability: {(1 - prediction):.2%}"
            )


# ---------------------------------
# Disclaimer
# ---------------------------------

st.divider()

st.caption(
    "This application is intended for educational and "
    "research purposes only and is not a medical diagnostic tool."
)