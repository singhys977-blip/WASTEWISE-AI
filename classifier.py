import streamlit as st
from PIL import Image
from transformers import pipeline


# ---------------------------------
# Waste labels
# ---------------------------------
WASTE_LABELS = [
    "plastic bottle",
    "plastic packaging",
    "paper",
    "cardboard",
    "glass bottle",
    "metal can",
    "food waste",
    "electronic waste",
    "hazardous waste",
    "other waste"
]


# ---------------------------------
# Load AI model only once
# ---------------------------------
@st.cache_resource
def load_classifier():

    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


# ---------------------------------
# Classify waste
# ---------------------------------
def classify_waste(image):

    classifier = load_classifier()

    # Support both uploaded PIL images
    # and image file paths
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    predictions = classifier(
        image,
        candidate_labels=WASTE_LABELS
    )

    best_prediction = predictions[0]

    return {
        "waste_type": best_prediction["label"],
        "confidence": best_prediction["score"],
        "predictions": predictions
    }



