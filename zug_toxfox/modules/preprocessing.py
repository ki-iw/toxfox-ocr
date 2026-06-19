import cv2
import numpy as np

from zug_toxfox import getLogger, pipeline_config

log = getLogger(__name__)


class PreProcessor:
    def __init__(self):
        self.steps = pipeline_config.preprocessing.steps
        # DotMap returns an empty DotMap for missing keys; fall back to a safe default.
        max_pixels = pipeline_config.preprocessing.max_pixels
        self.max_pixels = int(max_pixels) if isinstance(max_pixels, (int, float)) else 3_000_000

    def downscale(self, image: np.ndarray) -> np.ndarray:
        """Shrink oversized images (preserving aspect ratio) to bound OCR memory usage."""
        if not self.max_pixels:
            return image
        h, w = image.shape[:2]
        if h * w <= self.max_pixels:
            return image
        scale = (self.max_pixels / (h * w)) ** 0.5
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        log.info("Downscaling image from %sx%s to %sx%s", w, h, new_size[0], new_size[1])
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def get_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, 3)

    def thresholding(self, image: np.ndarray) -> np.ndarray:
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = self.downscale(image)
        if "gray" in self.steps and self.steps["gray"]["enabled"]:
            image = self.get_grayscale(image)
        if "blur" in self.steps and self.steps["blur"]["enabled"]:
            image = self.remove_noise(image)
        if "threshold" in self.steps and self.steps["threshold"]["enabled"]:
            image = self.thresholding(image)
        return image
