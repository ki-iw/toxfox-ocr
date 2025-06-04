# type: ignore
"""
ToxFox ocr pipeline
Usage:
    pipeline.py [--image_folder=./data/croped_inputs] [--output_path=./data/output] [--inci_path=./data/inci/inci.json] [--config_path=./zug_toxfox/pipeline_config.yaml] [--debug=True]
    pipeline.py (-h | --help)
Options:
    --config_path=<path>     Path to the configuration YAML file [default: ./zug_toxfox/pipeline_config.yaml]
    --debug=<bool>           Enable debug mode [default: True]
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import torch

from zug_toxfox import default_config, getLogger, pipeline_config
from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.ocr import OCR
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor
from zug_toxfox.modules.preprocessing import PreProcessor
from zug_toxfox.utils import create_output_folder, get_image_paths, save_run_info, str2bool

log = getLogger(__name__)


class Pipeline:
    def __init__(self, indexer: FAISSIndexer, evaluation: bool = False):
        self.preprocessor = PreProcessor()
        self.ocr = OCR()
        self.postprocessor = PostProcessor(indexer)
        self.evaluation = Evaluation() if evaluation is True else None

    def process_image(self, image):
        log.info("Preprocessing image...")
        processed_image = self.preprocessor.preprocess_image(image)
        log.info("Detecting ingredients...")
        detected_ingredients = self.ocr.process_image(processed_image, debug=False)  # type: ignore
        log.info("Postprocessing detected ingredients")
        result = self.postprocessor.get_ingredients(detected_ingredients)  # type: ignore
        return result

    def process_image_path(self, image_path: str, output_folder: str, debug: bool) -> None:
        image = cv2.imread(image_path)

        log.info("Processing image %s", image_path)

        result = self.process_image(image)

        # TODO: Fix for debug mode
        # processed_image = self.preprocessor.preprocess_image(image)
        # if debug:
        #     image_bb, detected_ingredients = self.ocr.process_image(processed_image, image, debug)
        # else:
        #     detected_ingredients = self.ocr.process_image(processed_image, debug)  # type: ignore
        # corrected_ingredients = self.postprocessor.get_ingredients(detected_ingredients)  # type: ignore

        image_output_path = create_output_folder(output_folder, f"{Path(image_path).stem}")

        log.info("Saving result to %s", image_output_path)
        with open(os.path.join(image_output_path, "ocr.txt"), "w") as text_file:
            text_file.write("INGREDIENTS:\n{}\n".format("\n".join(result.get("ingredients", "None found"))))
            text_file.write("\n---\n")
            text_file.write("POLLUTANS:\n{}\n".format("\n".join(result.get("pollutants", "None found"))))

        if debug:
            log.info("Debug %s: Saving ingredients box at %s", debug, image_output_path)
            # Fix for debug mode
            # cv2.imwrite(os.path.join(image_output_path, "bb_image.png"), image_bb)  # type: ignore

        if self.evaluation is not None:
            self.evaluation.evaluate(result["ingredients"], Path(image_path).stem, self.evaluation_method)

        log.info("Saving result to %s", image_output_path)
        with open(os.path.join(image_output_path, "ocr.txt"), "w") as text_file:
            text_file.write(",\n".join(result["ingredients"]))

    def main(self, debug: bool = False) -> None:
        run_id = datetime.now().strftime("run_%d.%m.%Y_%H:%M:%S")
        log.info(f"Starting run {run_id}")
        output_folder = create_output_folder(default_config.output_path, run_id)

        if not os.path.exists(default_config.image_path):
            log.warning("No images found in %s, break", default_config.image_path)
        else:
            image_paths = get_image_paths(default_config.image_path)

            for i, image_path in enumerate(image_paths):
                log.info("Processing image %s / %s, %s", i + 1, len(image_path), image_path)
                self.process_image_path(image_path, output_folder, debug)

            if self.evaluation is not None:
                accuracy, f1 = self.evaluation.get_final_metrics()
                log.info(
                    "(Method %s) Final Accuracy: %s, F1 Score: %s",
                    self.evaluation.method,
                    accuracy,
                    f1,
                )

            if debug:
                save_run_info(pipeline_config, self.evaluation.method, output_folder, run_id, accuracy, f1)
                log.info(f"Max memory usage: {torch.cuda.max_memory_allocated()/1000000000:.1f}GB")
            log.info(f"Run {run_id} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Pipeline")
    parser.add_argument("--config_path", type=str, default="zug_toxfox/pipeline_config.yaml")
    parser.add_argument("--debug", type=str2bool, default=True)

    args = parser.parse_args()

    log.info(
        "Processing images from %s and saving output to %s, debug=%s", args.image_folder, args.output_path, args.debug
    )
    pipeline = Pipeline(args.image_folder, args.output_path, args.config_path)
    pipeline.main(debug=args.debug)
