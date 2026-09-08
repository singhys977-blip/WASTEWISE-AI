import json

import streamlit as st
from PIL import Image

from classifier import classify_waste


# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="wide"
)


# ---------------------------------
# Load Waste Management Rules
# ---------------------------------
with open("waste_rules.json", "r", encoding="utf-8") as file:
    waste_rules = json.load(file)


# ---------------------------------
# Application Header
# ---------------------------------
st.title("♻️ Smart Waste Management AI")

st.subheader(
    "AI-powered waste classification and disposal guidance"
)

st.write(
    "Upload an image of waste and the AI will identify the "
    "waste type and suggest an appropriate handling method."
)


# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "📷 Upload a waste image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------
# Analyze Image
# ---------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste",
        width=400
    )

    st.success("Image uploaded successfully!")

    st.divider()

    # ---------------------------------
    # Run AI Classification
    # ---------------------------------
    with st.spinner("🤖 AI is analyzing the waste image..."):

        try:
            result = classify_waste(image)

        except Exception as error:
            st.error("❌ The AI could not analyze this image.")
            st.exception(error)
            st.stop()

    waste_type = result["waste_type"]
    confidence = result["confidence"]


    # ---------------------------------
    # Convert AI Result to Category
    # ---------------------------------
    if "food" in waste_type:
        category = "biodegradable"

    elif "electronic" in waste_type:
        category = "electronic"

    elif "hazardous" in waste_type:
        category = "hazardous"

    elif (
        "plastic" in waste_type
        or "paper" in waste_type
        or "cardboard" in waste_type
        or "glass" in waste_type
        or "metal" in waste_type
    ):
        category = "recyclable"

    else:
        category = "other"


    # ---------------------------------
    # Display AI Analysis
    # ---------------------------------
    st.subheader("🔍 AI Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            "**Detected Waste:**",
            waste_type.title()
        )

        st.write(
            "**Category:**",
            category.replace("_", " ").title()
        )

    with col2:
        st.write(
            "**AI Confidence:**",
            f"{confidence * 100:.2f}%"
        )

        if confidence < 0.50:
            st.warning(
                "⚠️ The AI is not very confident. "
                "Try uploading a clearer image."
            )

        elif confidence < 0.70:
            st.info(
                "ℹ️ The AI has moderate confidence. "
                "Please verify the waste type before disposal."
            )

        else:
            st.success(
                "✅ The AI has relatively high confidence."
            )


    # ---------------------------------
    # Waste Management Recommendation
    # ---------------------------------
    rule = waste_rules.get(category)

    if rule:

        st.subheader("♻️ Recommended Handling")

        st.info(rule["handling"])

        st.subheader("✅ Recommended Action")

        st.success(rule["recommended_action"])

        st.subheader("📚 Source")

        st.caption(rule["source"])

    else:

        st.warning(
            "No waste-management rule was found for this category."
        )


    # ---------------------------------
    # Detailed AI Predictions
    # ---------------------------------
    with st.expander("🔎 View AI Predictions"):

        for prediction in result["predictions"]:

            label = prediction["label"]
            score = prediction["score"]

            st.write(
                f"**{label.title()}** — "
                f"{score * 100:.2f}%"
            )


