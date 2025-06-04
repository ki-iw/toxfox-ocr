import os

import numpy as np
import yaml
from fuzzywuzzy import fuzz

from zug_toxfox import default_config, getLogger, pipeline_config

log = getLogger(__name__)


class Evaluation:
    def __init__(
        self, threshold: float = pipeline_config.evaluation.threshold, method: str = pipeline_config.evaluation.method
    ):
        self.ground_truth_path = default_config.evaluation.ground_truth_path
        self.method = method
        self.threshold = threshold
        self.acc_levenshtein: list[float] = []
        self.acc_exact: list[float] = []
        self.f1_levenshtein: list[float] = []
        self.f1_exact: list[float] = []

    def exact_match(self, prediction: list[str], ground_truth: list[str]) -> list[str]:
        gt_set = set(ground_truth)
        return [pred for pred in prediction if pred in gt_set]

    def levenshtein_match(self, prediction: list[str], ground_truth: list[str]) -> list[str]:
        matches = []
        for pred in prediction:
            best_match = max(ground_truth, key=lambda gt: fuzz.ratio(pred, gt))
            score = fuzz.ratio(pred, best_match.lower().strip())

            if score >= self.threshold:
                matches.append(pred)
        return matches

    def get_metrics(self, prediction: list[str], ground_truth: list[str], method: str = "exact"):
        if method == "exact":
            tp_list = self.exact_match(prediction, ground_truth)  # true positives
        elif method == "levenshtein":
            tp_list = self.levenshtein_match(prediction, ground_truth)  # true positives
        else:
            raise ValueError("Method not implemented")  # noqa: TRY003

        tp = len(tp_list)
        fp = len([pred for pred in prediction if pred not in set(tp_list)])  # false positives
        fn = len([gt for gt in ground_truth if gt not in set(prediction)])  # false negatives

        log.debug(f"False negatives: {[gt for gt in ground_truth if gt not in set(prediction)]}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        tp_rate = tp / len(ground_truth)  # true positive rate
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # F1 score
        return f1, tp_rate

    def evaluate(self, prediction: list[str], image_name: str, method: str = "exact"):
        try:
            file_path = os.path.join(self.ground_truth_path, f"{image_name}.yaml")
            with open(file_path) as file:
                data = yaml.safe_load(file)
            ground_truth = data.get("INCI_list")
        except OSError as e:
            log.exception("Error accessing file %s: %s", file_path, e)  # noqa: TRY401

        ground_truth = sorted([gt.lower().strip().replace("*", "") for gt in ground_truth])
        prediction = sorted([pred.lower().strip() for pred in prediction])

        # Metrics
        if method == "exact":
            f1_exact, tp_rate_exact = self.get_metrics(prediction, ground_truth, method)
            self.acc_exact.append(tp_rate_exact)
            self.f1_exact.append(f1_exact)

        elif method == "levenshtein":
            f1_levenshtein, tp_rate_levenshtein = self.get_metrics(prediction, ground_truth, method)
            self.acc_levenshtein.append(tp_rate_levenshtein)
            self.f1_levenshtein.append(f1_levenshtein)

    def get_final_metric(self):
        if self.evaluation_method == "exact":
            accuracy, f1 = (
                round(np.mean(self.evaluation.acc_levenshtein), 2),
                round(np.mean(self.evaluation.f1_levenshtein), 2),
            )
        else:
            accuracy, f1 = (
                round(np.mean(self.evaluation.acc_levenshtein), 2),
                round(np.mean(self.evaluation.f1_levenshtein), 2),
            )
        return accuracy, f1
