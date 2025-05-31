
import yaml
import logging
import torch
from pathlib import Path

# Set up logging configurations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set up device for inference
if torch.cuda.is_available():
    DEVICE = "auto"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Get the location of all data
DATA_PATH = Path(__file__).joinpath("..", "..", "data").resolve()

# Get the location of all model prompts
PROMPTS_PATH = Path(__file__).joinpath("..", "..", "prompts").resolve()

# Get pipeline configurations
PIPELINE_CONFIG_PATH = Path(__file__).joinpath("..", "..", "configs", "pipeline_configs.yaml").resolve()
PIPELINE_CONFIG = yaml.safe_load(open(PIPELINE_CONFIG_PATH, "r"))

# Get models configurations
MODELS_CONFIG_PATH = Path(__file__).joinpath("..", "..", "configs", "models_configs.yaml").resolve()
MODELS_CONFIG = yaml.safe_load(open(MODELS_CONFIG_PATH, "r"))

# Get Streamlit configurations
STREAMLIT_CONFIG_PATH = Path(__file__).joinpath("..", "..", "configs", "streamlit_configs.yaml").resolve()
STREAMLIT_CONFIG = yaml.safe_load(open(STREAMLIT_CONFIG_PATH, "r"))
