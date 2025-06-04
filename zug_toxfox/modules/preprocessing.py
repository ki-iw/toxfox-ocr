import cv2
import numpy as np

from zug_toxfox import getLogger, pipeline_config

log = getLogger(__name__)


class PreProcessor:
    def __init__(self):
        self.steps = pipeline_config.preprocessing.steps

    def get_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, 3)

    def thresholding(self, image: np.ndarray) -> np.ndarray:
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if "gray" in self.steps and self.steps["gray"]["enabled"]:
            image = self.get_grayscale(image)
        if "threshold" in self.steps and self.steps["threshold"]["enabled"]:
            image = self.thresholding(image)
        if "noise" in self.steps and self.steps["noise"]["enabled"]:
            image = self.remove_noise(image)
        return image
