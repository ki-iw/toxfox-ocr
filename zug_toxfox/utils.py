import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from zug_toxfox import default_config


def load_json(path: str) -> list[str]:
    with open(path) as file:
        return json.load(file)


def remove_duplicates(lst: list[str]) -> list[str]:
    return list(dict.fromkeys(lst))


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")  # noqa: TRY003


def process_pollutants() -> None:
    input_path = Path(default_config.pollutants_path)
    yaml_path = Path(default_config.synonym_path)
    json_path = Path(default_config.pollutants_path_simple)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {str(input_path)}")  # noqa: TRY003, RUF010

    if yaml_path.exists() and json_path.exists():
        return None

    df = pd.read_excel(input_path)

    ingredients = df.iloc[:, 1]
    synonyms = df.iloc[:, 2]

    synonym_to_ingredient = {}
    all_synonyms = []

    for ingredient, synonym_list_raw in zip(ingredients, synonyms):  # noqa: B905
        if pd.notna(synonym_list_raw):
            synonym_list = synonym_list_raw.split("\n")
            all_synonyms.extend(synonym_list)
            for synonym in synonym_list:
                synonym_to_ingredient[synonym.strip().lower()] = ingredient

    if not yaml_path.exists():
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(synonym_to_ingredient, yaml_file, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if not json_path.exists():
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(all_synonyms, json_file, indent=2, ensure_ascii=False)


def get_image_paths(image_folder: str, extensions=[".jpg", ".png", ".jpeg"]) -> list:  # noqa: B006
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def create_output_folder(output_path, folder_name: str) -> str:
    output_folder = os.path.join(output_path, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def save_run_info(pipeline_config, method, output_folder: str, run_id: str, accuracy: float, f1: float) -> None:
    info = {
        "config": pipeline_config,
        "run_info": {
            "run_id": run_id,
            "Evaluation": {
                "method": method,
                "accuracy": accuracy,
                "f1_score": f1,
            },
        },
    }
    config_path = os.path.join(output_folder, "run_info.yml")
    with open(config_path, "w") as outfile:
        yaml.dump(info, outfile, default_flow_style=False)
