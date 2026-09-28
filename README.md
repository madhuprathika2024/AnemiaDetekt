
# AnemiaDetekt

AI-Based Non-Invasive Anemia Screening
Using Conjunctiva Images

## Overview
An experimental image-classification application
using MobileNetV2 and Streamlit.

## Technologies
- Python
- TensorFlow
- MobileNetV2
- Streamlit
- NumPy
- Pillow

## Features
- Upload conjunctiva images
- Image preprocessing
- Anemia classification
- Display model output

## Current Model
Architecture: MobileNetV2
Input size: 224 x 224
Classes: Anemic, Non-anemic

Current test accuracy: 68.81%
Test images: 109

These are experimental results, not evidence
of clinical accuracy.

## Run locally

1. Create and activate a Python 3.11
   virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Run:

   streamlit run app.py

## Disclaimer
Research prototype only. Not a medical
diagnostic tool. Laboratory testing is
required to assess anemia.