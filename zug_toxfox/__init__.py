import os
from importlib import metadata
from importlib.metadata import version

import torch
import yaml
from dotenv import load_dotenv
from dotmap import DotMap

from .logger import getLogger

load_dotenv()

log = getLogger(__name__)

# Import configs
base_path = os.getenv("BASE_PATH", "/mnt/data/ZUG-ToxFox")
with open(os.path.join(os.path.dirname(__file__), "../config/default_config.yml")) as f:
    default_config = DotMap(
        {key: os.path.join(base_path, value.lstrip("/")) for key, value in yaml.safe_load(f).items()}
    )

with open(os.path.join(os.path.dirname(__file__), "../config/pipeline_config.yml")) as f:
    pipeline_config = DotMap(yaml.safe_load(f))

if torch.cuda.is_available():
    free_memory, total_memory = torch.cuda.mem_get_info()
    log.info("Running on CUDA!")
    log.info(f"Total Memory: {total_memory/1000000000:.1f}GB, free memory: {free_memory/1000000000:.1f}GB")
elif torch.backends.mps.is_available():
    log.info("Running on CUDA with MPS!")
else:
    log.info("Running on CPU!")

try:
    __version__ = version("zug_toxfox")
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


__all__ = [
    "BaseClass",
    "getLogger",
]
