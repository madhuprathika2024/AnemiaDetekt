
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AnemiaDetekt",
    page_icon="🩺",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f8fc;
}
h1 {
    color: #17365d;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #53657c;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path(__file__).parent / "best_mobilenetv2_v2.keras"

@st.cache_resource
def load_anemia_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------- HEADER ----------------
st.title("🩺 AnemiaDetekt")
st.markdown(
    '<p class="subtitle">AI-Based Non-Invasive Anemia Screening</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- UPLOAD IMAGE ----------------
st.subheader("Upload Conjunctiva Image")
st.write("Upload an image of the inner lower eyelid.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- DETECTION ----------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

        st.success("Image uploaded successfully!")

        if st.button("Detect Anemia", use_container_width=True):
            
            try:
                with st.spinner("Analyzing image..."):

                    # Resize image to model input size
                    img = image.resize((224, 224))

                    # Convert image to NumPy array
                    img_array = np.array(
                        img, dtype=np.float32
                    )

                    # Add batch dimension
                    img_array = np.expand_dims(
                        img_array, axis=0
                    )

                    # Load trained model
                    model = load_anemia_model()

                    # Model already includes MobileNetV2
                    # preprocessing, so don't apply it again.
                    prediction = model.predict(
                        img_array, verbose=0
                    )

                    score = float(
                        np.asarray(prediction).reshape(-1)[0]
                    )

                # Binary sigmoid output
                st.subheader("Screening Result")

                if score >= 0.5:
                    st.success("Model prediction: Non-anemic")
                else:
                    st.warning("Model prediction: Anemic")

                st.write(f"Raw model output: {score:.4f}")

                st.caption(
                    "This is an experimental model prediction, "
                    "not a medical diagnosis. Laboratory testing "
                    "is required to assess anemia."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    except Exception as e:
        st.error(f"Unable to read image: {e}")

else:
    st.info("Please upload an image to continue.")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "AnemiaDetekt | Research prototype | "
    "AI-based image screening"
)