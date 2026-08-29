import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection using VGG16",
    page_icon="🫁",
    layout="centered"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_URL = (
    "https://huggingface.co/dikshaballav/"
    "pneumonia-vgg16/resolve/main/"
    "pneumonia_vgg16.weights.h5"
)

WEIGHTS_PATH = "pneumonia_vgg16.weights.h5"


# ============================================================
# DOWNLOAD MODEL WEIGHTS
# ============================================================

def download_weights():

    if os.path.exists(WEIGHTS_PATH):
        return

    st.info("Downloading trained model weights...")

    try:
        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=600
        )

        response.raise_for_status()

        total_size = int(
            response.headers.get("content-length", 0)
        )

        downloaded = 0

        progress_bar = st.progress(0)

        with open(WEIGHTS_PATH, "wb") as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:

                        progress = int(
                            downloaded / total_size * 100
                        )

                        progress_bar.progress(
                            min(progress, 100)
                        )

        progress_bar.empty()

        st.success("Model weights downloaded successfully.")

    except Exception as e:

        if os.path.exists(WEIGHTS_PATH):
            os.remove(WEIGHTS_PATH)

        st.error("Unable to download the model weights.")

        st.exception(e)

        st.stop()


# ============================================================
# CREATE AND LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Download trained weights if necessary
    download_weights()

    # --------------------------------------------------------
    # VGG16 base model
    # --------------------------------------------------------

    base = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze VGG16 layers
    for layer in base.layers:
        layer.trainable = False

    # --------------------------------------------------------
    # Classification model
    # --------------------------------------------------------

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    # --------------------------------------------------------
    # Load your trained weights
    # --------------------------------------------------------

    model.load_weights(WEIGHTS_PATH)

    return model


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading pneumonia detection model..."):

    model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🫁 Pneumonia Detection using VGG16")

st.write(
    "Upload a chest X-ray image and the trained deep learning "
    "model will classify it as Normal or Pneumonia."
)

st.warning(
    "⚠️ This application is for educational and research "
    "purposes only. It is not a medical diagnostic tool."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# IMAGE PROCESSING AND PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display original image
    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    # Prediction button
    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        with st.spinner("Analyzing X-ray..."):

            # ------------------------------------------------
            # Resize image
            # ------------------------------------------------

            image_resized = image.resize(
                (224, 224)
            )

            # ------------------------------------------------
            # Convert image to NumPy array
            # ------------------------------------------------

            image_array = np.array(
                image_resized
            )

            # ------------------------------------------------
            # Normalize pixel values
            # ------------------------------------------------

            image_array = image_array / 255.0

            # ------------------------------------------------
            # Add batch dimension
            # ------------------------------------------------

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # ------------------------------------------------
            # Make prediction
            # ------------------------------------------------

            prediction = model.predict(
                image_array,
                verbose=0
            )[0][0]

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        # Your training class mapping:
        #
        # NORMAL = 0
        # PNEUMONIA = 1

        pneumonia_probability = float(
            prediction
        )

        normal_probability = 1.0 - pneumonia_probability


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader("Prediction Result")


        if pneumonia_probability >= 0.5:

            st.error(
                "⚠️ Pneumonia Detected"
            )

        else:

            st.success(
                "✅ Normal"
            )


        # ----------------------------------------------------
        # Display probabilities
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Normal",
                f"{normal_probability:.2%}"
            )

        with col2:

            st.metric(
                "Pneumonia",
                f"{pneumonia_probability:.2%}"
            )


        # ----------------------------------------------------
        # Probability bar
        # ----------------------------------------------------

        st.write(
            "Pneumonia probability"
        )

        st.progress(
            min(
                max(
                    pneumonia_probability,
                    0.0
                ),
                1.0
            )
        )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.subheader("About this project")

st.write(
    "This application uses a VGG16-based transfer learning "
    "model for binary classification of chest X-ray images."
)

st.write(
    "**Classes:** Normal and Pneumonia"
)

st.write(
    "**Input size:** 224 × 224 pixels"
)

st.write(
    "**Model:** VGG16 + Flatten + Dense(256) + "
    "Dropout(0.5) + Dense(1, sigmoid)"
)

st.caption(
    "Educational/research demonstration only — not for "
    "clinical diagnosis."
)
