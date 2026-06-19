import os

# Bound CPU thread pools BEFORE torch/numpy are imported. On CPU-only hosts torch
# otherwise spawns one worker per core, and the transient OCR buffers can push peak
# RSS high enough to get OOM-killed on a small machine. Env vars (set e.g. by
# run_api.sh) take precedence; these are just safe defaults. set_num_threads below
# enforces it at runtime too, since OpenMP reads these only at import time.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    # Enforce the thread cap at runtime (env vars only affect OpenMP at import time).
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    log.info("Running on CPU! (torch threads capped at %d)", torch.get_num_threads())

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
