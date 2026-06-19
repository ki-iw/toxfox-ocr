import asyncio
import os
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

# TODO: mypy
from zug_toxfox import default_config, getLogger
from zug_toxfox.inci_processor import ProcessInci  # type: ignore[attr-defined]
from zug_toxfox.modules.postprocessing import FAISSIndexer  # type: ignore[attr-defined]
from zug_toxfox.pipeline import Pipeline  # type: ignore[attr-defined]
from zug_toxfox.utils import process_pollutants

log = getLogger(__name__)


inci_processor = ProcessInci(inci_xlsx_path=default_config.inci_path, output_path=default_config.inci_path_simple)

update_event = threading.Event()

# Serialize the heavy OCR step so only one image is ever in flight at a time. Combined
# with the per-image downscale cap (config/pipeline_config.yml: max_pixels), this bounds
# peak RSS to ~3.2 GB regardless of how many requests arrive concurrently or whether the
# server is later run with multiple threads/workers — keeping the process well under 8 GB.
ocr_semaphore = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI, reload_data: bool = False):
    log.info("Starting up application..")
    log.info("Processing inci list %s", default_config.inci_path)

    update_event.set()

    try:
        faiss_path = default_config.faiss_path
        if reload_data or not os.path.exists(faiss_path) or not os.listdir(faiss_path):
            inci_processor.get_inci_json()
            process_pollutants()
            app.state.indexer = FAISSIndexer()
        else:
            app.state.indexer = FAISSIndexer()
    finally:
        update_event.clear()

    yield


def get_pipeline() -> Pipeline:
    if update_event.is_set():
        log.info("Waiting for data update to complete...")
        update_event.wait()
    indexer = app.state.indexer
    if indexer is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Pipeline(indexer=indexer, evaluation=False)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok"})


@app.post("/process_image")
async def process_image(file: UploadFile = File(...), pipeline=Depends(get_pipeline)):
    """
    Extract ocr text from images
    """
    # Read the uploaded file into a numpy array
    file_bytes = np.frombuffer(file.file.read(), np.uint8)
    # Decode the numpy array into an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    log.info("Processing image %s", file.filename)

    try:
        # Single-flight: serialize OCR and run it off the event loop so /health stays
        # responsive. Bounds concurrent memory use (see ocr_semaphore note above).
        async with ocr_semaphore:
            result = await asyncio.to_thread(pipeline.process_image, image)
        response = {
            "image_name": file.filename,
            "ingredients": [result["ingredients"]],
            "pollutants": [result["pollutants"]],
        }
        return JSONResponse(status_code=status.HTTP_200_OK, content=response)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)  # noqa: B904, TRY200


@app.post("/update_data")
async def update_data():
    """
    Update the INCI data, pollutants, and rebuild the FAISS index.
    """
    update_event.set()

    try:
        inci_processor.get_inci_json()
        process_pollutants()
        app.state.indexer = FAISSIndexer()
    finally:
        update_event.clear()

    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "Data updated"})


if __name__ == "__main__":
    # Bind address is configurable via env (HOST/PORT). Defaults to localhost; set
    # HOST to 0.0.0.0 (or a specific IP) to expose the service on the network.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8502"))
    log.info("Starting uvicorn on http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
