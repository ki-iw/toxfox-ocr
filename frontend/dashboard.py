import cv2
import numpy as np
import streamlit as st

from zug_toxfox import getLogger
from zug_toxfox.pipeline import Pipeline

log = getLogger(__name__)

pipeline = Pipeline(
    image_folder="", output_path="", inci_path="data/inci/inci.json", config_path="zug_toxfox/pipeline_config.yaml"
)

# Set the title of the app
st.title("ToxFox Dashboard")

st.write("Upload a photo of an ingredient list to get started!")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Add a submit button
    if st.button("Submit"):
        with st.spinner("Processing image..."):
            # Processing image here
            processed_image = pipeline.preprocessor.preprocess_image(image)
            detected_ingredients = pipeline.ocr.process_image(processed_image)

            log.info("Detected Ingredients: (%s), %s", len(detected_ingredients), detected_ingredients)

            corrected_ingredients = pipeline.postprocessor.get_ingredients(detected_ingredients)

            log.info("Corrected Ingredients: (%s), %s", len(corrected_ingredients), corrected_ingredients)
            st.write(f"Detected Ingredients: {corrected_ingredients}")
