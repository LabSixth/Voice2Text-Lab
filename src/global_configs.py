
import yaml
import logging
from pathlib import Path

# Set up logging configurations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the location of all data
DATA_PATH = Path(__file__).joinpath("..", "..", "data").resolve()

# Get pipeline configurations
PIPELINE_CONFIG_PATH = Path(__file__).joinpath("..", "..", "configs", "pipeline_configs.yaml").resolve()
PIPELINE_CONFIG = yaml.safe_load(open(PIPELINE_CONFIG_PATH, "r"))
