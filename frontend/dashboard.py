import sys
from pathlib import Path

# `streamlit run frontend/dashboard.py` puts this file's folder (frontend/) on sys.path,
# not the project root where the zug_toxfox package lives. Add the project root so the
# import below resolves regardless of the directory the app is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st

from zug_toxfox import getLogger
from zug_toxfox.modules.postprocessing import FAISSIndexer
from zug_toxfox.pipeline import Pipeline

log = getLogger(__name__)


@st.cache_resource
def get_pipeline() -> Pipeline:
    """Build the pipeline once and cache it across reruns."""
    indexer = FAISSIndexer()
    return Pipeline(indexer=indexer, evaluation=False)


pipeline = get_pipeline()

# Set the title of the app
st.title("ToxFox Dashboard")

st.write("Upload a photo of an ingredient list to get started!")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the image (convert from BGR to RGB for correct colors)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image.", width="stretch")

    # Add a submit button
    if st.button("Submit"):
        with st.spinner("Processing image..."):
            result = pipeline.process_image(image)

            log.info("Result: %s", result)

            st.subheader("Ingredients")
            st.write(result["ingredients"] or "None found")

            st.subheader("Pollutants")
            st.write(result["pollutants"] or "None found")
